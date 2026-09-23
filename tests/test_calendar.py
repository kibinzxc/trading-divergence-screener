"""Tests for the news-calendar feature (JBlanked primary, ForexFactory fallback).

Run with:  python -m unittest tests.test_calendar -v
"""
import datetime as dt
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402


# One row from each upstream, shaped exactly as the feeds serve them.
FF_ROW = {"title": "CPI m/m", "country": "USD", "date": "2026-09-24T08:30:00-04:00",
          "impact": "High", "forecast": "0.3%", "previous": "0.2%"}
JB_ROW = {"Name": "Core CPI m/m", "Currency": "USD", "Category": "Inflation",
          "Impact": "High", "Date": "2024.02.08 15:30:00",
          "Actual": 0.4, "Forecast": 0.4, "Previous": 0.2,
          "Outcome": "Data Better Than Expected", "Strength": "Strong Data",
          "Quality": "Good Data"}


class DotEnv(unittest.TestCase):
    def setUp(self):
        os.environ.pop("JBLANKED_API_KEY", None)

    def _write(self, text):
        d = tempfile.mkdtemp()
        p = os.path.join(d, ".env")
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_lowercase_key_in_file_becomes_uppercase_env_var(self):
        p = self._write("# comment\njblanked_api_key = 'abc123'\n\nOTHER=x\n")
        m.load_dotenv(p)
        self.assertEqual(os.environ["JBLANKED_API_KEY"], "abc123")
        self.assertEqual(m.jb_api_key(), "abc123")

    def test_existing_environment_wins_over_file(self):
        os.environ["JBLANKED_API_KEY"] = "fromenv"
        p = self._write("jblanked_api_key=fromfile\n")
        m.load_dotenv(p)
        self.assertEqual(m.jb_api_key(), "fromenv")

    def test_missing_file_loads_nothing(self):
        self.assertEqual(m.load_dotenv(os.path.join(tempfile.mkdtemp(), "nope")), 0)
        self.assertEqual(m.jb_api_key(), "")


class MoverFilter(unittest.TestCase):
    def test_us_inflation_jobs_fed_and_growth_prints_are_movers(self):
        for title in ["CPI m/m", "Core CPI m/m", "CPI y/y", "Core PCE Price Index m/m",
                      "PPI m/m", "Core PPI m/m", "Non-Farm Employment Change",
                      "Unemployment Rate", "Average Hourly Earnings m/m",
                      "Federal Funds Rate", "FOMC Statement", "FOMC Press Conference",
                      "FOMC Meeting Minutes", "FOMC Economic Projections",
                      "Fed Chair Powell Speaks", "Fed Chair Powell Testifies",
                      "Advance GDP q/q", "Prelim GDP q/q", "Final GDP q/q",
                      "Retail Sales m/m", "Core Retail Sales m/m",
                      "ISM Manufacturing PMI", "ISM Services PMI"]:
            self.assertTrue(m.is_mover("USD", title), title)

    def test_foreign_central_bank_rate_decisions_are_movers(self):
        self.assertTrue(m.is_mover("JPY", "BOJ Policy Rate"))
        self.assertTrue(m.is_mover("EUR", "Main Refinancing Rate"))
        self.assertTrue(m.is_mover("GBP", "Official Bank Rate"))

    def test_scheduled_noise_is_not_a_mover(self):
        for cur, title in [("USD", "Unemployment Claims"), ("USD", "JOLTS Job Openings"),
                           ("USD", "CB Consumer Confidence"), ("USD", "Flash Manufacturing PMI"),
                           ("USD", "FOMC Member Waller Speaks"), ("USD", "Bank Holiday"),
                           ("AUD", "Unemployment Rate"), ("AUD", "RBA Gov Bullock Speaks"),
                           ("EUR", "CPI Flash Estimate y/y"), ("GBP", "CPI y/y"),
                           ("CAD", "Retail Sales m/m"), ("JPY", "BOJ Gov Ueda Speaks")]:
            self.assertFalse(m.is_mover(cur, title), f"{cur} {title}")


class Normalise(unittest.TestCase):
    def test_forexfactory_row_is_converted_from_eastern_to_utc(self):
        ev = m.normalise_events([FF_ROW], "forexfactory")[0]
        self.assertEqual(ev["ts"], "2026-09-24T12:30:00Z")
        self.assertEqual(ev["currency"], "USD")
        self.assertEqual(ev["impact"], "High")
        self.assertEqual(ev["title"], "CPI m/m")
        self.assertEqual(ev["forecast"], "0.3%")
        self.assertEqual(ev["previous"], "0.2%")
        self.assertEqual(ev["actual"], "")
        self.assertTrue(ev["mover"])

    def test_jblanked_row_is_taken_as_utc_and_numbers_become_strings(self):
        ev = m.normalise_events([JB_ROW], "jblanked")[0]
        self.assertEqual(ev["ts"], "2024-02-08T15:30:00Z")
        self.assertEqual(ev["title"], "Core CPI m/m")
        self.assertEqual(ev["actual"], "0.4")
        self.assertEqual(ev["forecast"], "0.4")
        self.assertEqual(ev["previous"], "0.2")
        self.assertTrue(ev["mover"])

    def test_jblanked_nulls_become_empty_strings(self):
        row = dict(JB_ROW, Actual=None, Forecast=None, Previous=None)
        ev = m.normalise_events([row], "jblanked")[0]
        self.assertEqual((ev["actual"], ev["forecast"], ev["previous"]), ("", "", ""))

    def test_events_are_sorted_by_time_and_deduped(self):
        rows = [FF_ROW, dict(FF_ROW),
                dict(FF_ROW, title="PPI m/m", date="2026-09-23T08:30:00-04:00")]
        evs = m.normalise_events(rows, "forexfactory")
        self.assertEqual([e["title"] for e in evs], ["PPI m/m", "CPI m/m"])

    def test_rows_without_a_usable_date_are_skipped(self):
        rows = [dict(FF_ROW, date=""), dict(FF_ROW, date="not a date"), {"title": "x"}, FF_ROW]
        self.assertEqual(len(m.normalise_events(rows, "forexfactory")), 1)

    def test_non_mover_is_flagged_false(self):
        ev = m.normalise_events([dict(FF_ROW, title="Unemployment Claims")], "forexfactory")[0]
        self.assertFalse(ev["mover"])

    def test_jblanked_zero_numbers_mean_missing(self):
        # JBlanked fills unknown forecast/previous/actual with 0, not null.
        row = dict(JB_ROW, Actual=0, Forecast=0.0, Previous=0)
        ev = m.normalise_events([row], "jblanked")[0]
        self.assertEqual((ev["actual"], ev["forecast"], ev["previous"]), ("", "", ""))


class Merge(unittest.TestCase):
    """ForexFactory is authoritative for the week it serves; JBlanked (which
    drops events and runs on a broker clock) only extends the horizon."""

    def _ff(self, ts, title="CPI m/m", **kw):
        e = {"ts": ts, "currency": "USD", "impact": "High", "title": title,
             "forecast": "0.3%", "previous": "0.2%", "actual": "", "mover": True}
        e.update(kw)
        return e

    def _jb(self, ts, title="CPI m/m", **kw):
        e = {"ts": ts, "currency": "USD", "impact": "High", "title": title,
             "forecast": "", "previous": "", "actual": "", "mover": True}
        e.update(kw)
        return e

    def test_shared_event_keeps_forexfactory_time_and_values_but_takes_jblanked_actual(self):
        out, shift = m.merge_calendars([self._ff("2026-09-24T12:30:00Z")],
                                       [self._jb("2026-09-24T18:30:00Z", actual="0.4")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ts"], "2026-09-24T12:30:00Z")
        self.assertEqual(out[0]["forecast"], "0.3%")
        self.assertEqual(out[0]["actual"], "0.4")
        self.assertEqual(shift, 6.0)

    def test_jblanked_only_events_are_shifted_by_the_measured_offset(self):
        out, shift = m.merge_calendars(
            [self._ff("2026-09-24T12:30:00Z")],
            [self._jb("2026-09-24T18:30:00Z"),
             self._jb("2026-10-02T18:30:00Z", title="Non-Farm Employment Change")])
        nfp = [e for e in out if e["title"].startswith("Non-Farm")][0]
        self.assertEqual(nfp["ts"], "2026-10-02T12:30:00Z")
        self.assertEqual(shift, 6.0)

    def test_forexfactory_events_missing_from_jblanked_are_kept(self):
        out, _ = m.merge_calendars(
            [self._ff("2026-09-24T12:30:00Z"), self._ff("2026-09-23T18:00:00Z", title="Federal Funds Rate")],
            [self._jb("2026-09-24T18:30:00Z")])
        self.assertEqual(sorted(e["title"] for e in out), ["CPI m/m", "Federal Funds Rate"])

    def test_no_overlap_means_no_shift_and_everything_is_kept(self):
        out, shift = m.merge_calendars([self._ff("2026-09-23T12:30:00Z", title="PPI m/m")],
                                       [self._jb("2026-10-02T12:30:00Z", title="Non-Farm Employment Change")])
        self.assertIsNone(shift)
        self.assertEqual([e["ts"] for e in out], ["2026-09-23T12:30:00Z", "2026-10-02T12:30:00Z"])

    def test_repeated_title_is_not_duplicated_when_occurrences_are_close(self):
        """Two same-titled events a couple of hours apart (FOMC speakers, Buba
        speakers) must pair one-to-one, not collapse onto the nearest and leave
        the other to reappear as a JBlanked-only event."""
        ff = [self._ff("2026-09-24T13:00:00Z", title="FOMC Member Williams Speaks", mover=False),
              self._ff("2026-09-24T17:00:00Z", title="FOMC Member Williams Speaks", mover=False)]
        jb = [self._jb("2026-09-24T19:00:00Z", title="FOMC Member Williams Speaks", mover=False, actual="A"),
              self._jb("2026-09-24T23:00:00Z", title="FOMC Member Williams Speaks", mover=False, actual="B")]
        out, shift = m.merge_calendars(ff, jb)
        self.assertEqual(shift, 6.0)
        self.assertEqual([e["ts"] for e in out],
                         ["2026-09-24T13:00:00Z", "2026-09-24T17:00:00Z"])
        self.assertEqual([e["actual"] for e in out], ["A", "B"])

    def test_repeated_title_does_not_skew_the_measured_shift(self):
        """Nearest-neighbour matching lets the earlier JBlanked event pair with
        the later ForexFactory one, which drags the median off the true offset."""
        ff = [self._ff("2026-09-24T13:00:00Z", title="ECB President Lagarde Speaks", mover=False),
              self._ff("2026-09-24T17:00:00Z", title="ECB President Lagarde Speaks", mover=False)]
        jb = [self._jb("2026-09-24T16:00:00Z", title="ECB President Lagarde Speaks", mover=False),
              self._jb("2026-09-24T20:00:00Z", title="ECB President Lagarde Speaks", mover=False)]
        self.assertEqual(m.calendar_time_check(jb, ff), 3.0)

    def test_result_is_sorted_after_shifting(self):
        out, _ = m.merge_calendars(
            [self._ff("2026-09-24T12:30:00Z")],
            [self._jb("2026-09-24T18:30:00Z"),
             self._jb("2026-09-24T14:00:00Z", title="Fed Chair Powell Speaks")])   # really 08:00Z
        self.assertEqual([e["title"] for e in out], ["Fed Chair Powell Speaks", "CPI m/m"])


class TimeCheck(unittest.TestCase):
    """JBlanked's clock offset is documented loosely, so the ForexFactory feed
    (which carries explicit UTC offsets) is used to confirm the two agree."""

    def _ev(self, ts, title="CPI m/m", cur="USD"):
        return {"ts": ts, "title": title, "currency": cur}

    def test_zero_when_shared_events_agree(self):
        a = [self._ev("2026-09-24T12:30:00Z")]
        b = [self._ev("2026-09-24T12:30:00Z")]
        self.assertEqual(m.calendar_time_check(a, b), 0.0)

    def test_reports_the_median_hour_shift(self):
        a = [self._ev("2026-09-24T15:30:00Z"), self._ev("2026-09-25T15:30:00Z", "PPI m/m")]
        b = [self._ev("2026-09-24T12:30:00Z"), self._ev("2026-09-25T12:30:00Z", "PPI m/m")]
        self.assertEqual(m.calendar_time_check(a, b), 3.0)

    def test_none_when_nothing_overlaps(self):
        a = [self._ev("2026-09-24T12:30:00Z", "CPI m/m")]
        b = [self._ev("2026-09-24T12:30:00Z", "PPI m/m")]
        self.assertIsNone(m.calendar_time_check(a, b))


class Refresh(unittest.TestCase):
    def setUp(self):
        self._orig = (m.CAL_STATE_FILE, m.fetch_jblanked, m.fetch_ff)
        m.CAL_STATE_FILE = os.path.join(tempfile.mkdtemp(), "calendar_state.json")
        m._cal_state.update({"events": [], "ts": None, "attempt": None, "meta": {},
                             "jb": {"events": [], "ts": None, "attempt": None}})
        self.calls = {"jb": 0, "ff": 0}
        os.environ["JBLANKED_API_KEY"] = "k"

        def jb(key):
            self.calls["jb"] += 1
            return [JB_ROW]

        def ff():
            self.calls["ff"] += 1
            return [FF_ROW]

        m.fetch_jblanked, m.fetch_ff = jb, ff

    def tearDown(self):
        m.CAL_STATE_FILE, m.fetch_jblanked, m.fetch_ff = self._orig
        os.environ.pop("JBLANKED_API_KEY", None)

    def _state(self):
        with m._lock:
            return json.loads(json.dumps({"events": m._cal_state["events"],
                                          "meta": m._cal_state["meta"]}))

    def test_merges_both_feeds_when_a_key_is_present(self):
        m.refresh_calendar()
        s = self._state()
        self.assertEqual(s["meta"]["source"], "jblanked+forexfactory")
        self.assertEqual(sorted(e["title"] for e in s["events"]), ["CPI m/m", "Core CPI m/m"])

    def test_uses_jblanked_alone_when_forexfactory_is_down(self):
        def boom():
            raise RuntimeError("ff down")
        m.fetch_ff = boom
        m.refresh_calendar()
        s = self._state()
        self.assertEqual(s["meta"]["source"], "jblanked")
        self.assertEqual([e["title"] for e in s["events"]], ["Core CPI m/m"])
        self.assertIn("ff down", s["meta"]["error"])

    def test_jblanked_alone_still_applies_the_last_known_clock_correction(self):
        """Losing ForexFactory must not silently serve raw broker-clock times:
        the previously measured shift is reused and reported."""
        m.fetch_jblanked = lambda key: [dict(JB_ROW, Name="CPI m/m", Date="2026.09.24 18:30:00")]
        m.fetch_ff = lambda: [dict(FF_ROW, title="CPI m/m", date="2026-09-24T08:30:00-04:00")]
        m.refresh_calendar()
        self.assertEqual(self._state()["meta"]["time_check_h"], 6.0)

        def boom():
            raise RuntimeError("ff down")
        m.fetch_ff = boom
        m.fetch_jblanked = lambda key: [dict(JB_ROW, Name="Core PCE Price Index m/m",
                                             Date="2026.10.01 18:30:00")]
        m._cal_state["attempt"] -= dt.timedelta(seconds=m.CAL_MIN_REFRESH_S + 1)
        m._cal_state["jb"]["attempt"] -= dt.timedelta(minutes=m.CAL_JB_REFRESH_MIN + 1)
        m.refresh_calendar(force=True)
        s = self._state()
        self.assertEqual(s["meta"]["source"], "jblanked")
        self.assertEqual([e["ts"] for e in s["events"]], ["2026-10-01T12:30:00Z"])
        self.assertEqual(s["meta"]["time_check_h"], 6.0)
        self.assertIn("clock", (s["meta"]["fallback"] or "").lower())

    def test_coverage_end_is_the_requested_horizon_not_the_last_event(self):
        """An empty Saturday inside the published week is quiet, not unknown, so
        the horizon must come from the range asked for."""
        m.refresh_calendar()
        s = self._state()
        last_event_day = max(e["ts"][:10] for e in s["events"])
        self.assertGreater(s["meta"]["covered_to"], last_event_day)

    def test_forexfactory_only_coverage_reaches_the_end_of_its_week(self):
        os.environ.pop("JBLANKED_API_KEY")
        # A Wednesday event; ForexFactory's feed always runs Sunday to Saturday.
        m.fetch_ff = lambda: [dict(FF_ROW, date="2026-09-23T08:30:00-04:00")]
        m.refresh_calendar()
        self.assertEqual(self._state()["meta"]["covered_to"], "2026-09-26")

    def test_falls_back_to_forexfactory_without_a_key(self):
        os.environ.pop("JBLANKED_API_KEY")
        m.refresh_calendar()
        s = self._state()
        self.assertEqual(s["meta"]["source"], "forexfactory")
        self.assertEqual([e["title"] for e in s["events"]], ["CPI m/m"])
        self.assertEqual(self.calls["jb"], 0)

    def test_falls_back_and_records_error_when_jblanked_fails(self):
        def boom(key):
            raise RuntimeError("HTTP 401")
        m.fetch_jblanked = boom
        m.refresh_calendar()
        s = self._state()
        self.assertEqual(s["meta"]["source"], "forexfactory")
        self.assertIn("401", s["meta"]["error"])

    def test_keeps_last_good_events_when_every_upstream_fails(self):
        m.refresh_calendar()
        first_ts = m._cal_state["ts"]
        kept = [e["title"] for e in m._cal_state["events"]]

        def boom(*a):
            raise RuntimeError("down")
        m.fetch_jblanked, m.fetch_ff = boom, boom
        m._cal_state["attempt"] = None          # clear the cooldown
        m._cal_state["jb"]["attempt"] = None
        m.refresh_calendar(force=True)
        s = self._state()
        self.assertEqual([e["title"] for e in s["events"]], kept)
        self.assertIn("down", s["meta"]["error"])
        self.assertEqual(m._cal_state["ts"], first_ts)

    def test_second_call_inside_cooldown_does_not_hit_upstream(self):
        m.refresh_calendar()
        m.refresh_calendar()
        self.assertEqual(self.calls["jb"], 1)

    def test_force_respects_the_minimum_interval_then_refetches(self):
        m.refresh_calendar()
        m.refresh_calendar(force=True)              # too soon after the last attempt
        self.assertEqual(self.calls["ff"], 1)
        m._cal_state["attempt"] -= dt.timedelta(seconds=m.CAL_MIN_REFRESH_S + 1)
        m.refresh_calendar(force=True)
        self.assertEqual(self.calls["ff"], 2)

    def test_time_check_flags_a_clock_disagreement(self):
        # JBlanked says 15:30Z, ForexFactory says 12:30Z for the same print.
        m.fetch_jblanked = lambda key: [dict(JB_ROW, Name="CPI m/m", Date="2026.09.24 15:30:00")]
        m.refresh_calendar()
        self.assertEqual(self._state()["meta"]["time_check_h"], 3.0)

    def test_state_round_trips_through_disk(self):
        m.refresh_calendar()
        saved = [e["title"] for e in m._cal_state["events"]]
        m._cal_state.update({"events": [], "ts": None, "attempt": None, "meta": {}})
        m.load_calendar()
        s = self._state()
        self.assertEqual([e["title"] for e in s["events"]], saved)
        self.assertEqual(sorted(saved), ["CPI m/m", "Core CPI m/m"])
        self.assertIsInstance(m._cal_state["ts"], dt.datetime)


class JbCredits(unittest.TestCase):
    """JBlanked's calendar endpoints cost credits per call, while ForexFactory
    is free. JBlanked only supplies the schedule beyond the current week, which
    barely changes, so it is fetched on a much slower cadence than the feed that
    carries today's forecasts."""

    def setUp(self):
        self._orig = (m.CAL_STATE_FILE, m.fetch_jblanked, m.fetch_ff)
        m.CAL_STATE_FILE = os.path.join(tempfile.mkdtemp(), "calendar_state.json")
        m._cal_state.update({"events": [], "ts": None, "attempt": None, "meta": {},
                             "jb": {"events": [], "ts": None, "attempt": None}})
        self.calls = {"jb": 0, "ff": 0}
        os.environ["JBLANKED_API_KEY"] = "k"

        def jb(key):
            self.calls["jb"] += 1
            return [dict(JB_ROW, Name="Non-Farm Employment Change", Date="2026.10.02 12:30:00")]

        def ff():
            self.calls["ff"] += 1
            return [FF_ROW]

        m.fetch_jblanked, m.fetch_ff = jb, ff

    def tearDown(self):
        m.CAL_STATE_FILE, m.fetch_jblanked, m.fetch_ff = self._orig
        m._cal_state.pop("jb", None)
        os.environ.pop("JBLANKED_API_KEY", None)

    def _age_ff(self):
        """Let the cheap feed go stale without ageing the metered one."""
        m._cal_state["attempt"] -= dt.timedelta(minutes=m.CAL_REFRESH_MIN + 1)

    def test_forexfactory_refreshes_without_spending_a_jblanked_credit(self):
        m.refresh_calendar()
        self.assertEqual((self.calls["ff"], self.calls["jb"]), (1, 1))
        self._age_ff()
        m.refresh_calendar()
        self.assertEqual(self.calls["ff"], 2)
        self.assertEqual(self.calls["jb"], 1, "JBlanked must not be re-fetched while its cache is fresh")

    def test_cached_jblanked_events_still_extend_the_horizon(self):
        m.refresh_calendar()
        self._age_ff()
        m.refresh_calendar()
        titles = [e["title"] for e in m._cal_state["events"]]
        self.assertIn("Non-Farm Employment Change", titles)
        self.assertIn("CPI m/m", titles)

    def test_jblanked_is_refetched_once_its_own_interval_passes(self):
        m.refresh_calendar()
        self._age_ff()
        m._cal_state["jb"]["attempt"] -= dt.timedelta(minutes=m.CAL_JB_REFRESH_MIN + 1)
        m.refresh_calendar()
        self.assertEqual(self.calls["jb"], 2)

    def test_a_failed_jblanked_fetch_keeps_the_cached_horizon(self):
        m.refresh_calendar()

        def boom(key):
            raise RuntimeError("HTTP 401: no credits")
        m.fetch_jblanked = boom
        self._age_ff()
        m._cal_state["jb"]["attempt"] -= dt.timedelta(minutes=m.CAL_JB_REFRESH_MIN + 1)
        m.refresh_calendar()
        titles = [e["title"] for e in m._cal_state["events"]]
        self.assertIn("Non-Farm Employment Change", titles)
        self.assertIn("credits", m._cal_state["meta"]["error"])

    def test_jblanked_cache_round_trips_through_disk(self):
        m.refresh_calendar()
        m._cal_state.update({"events": [], "ts": None, "attempt": None, "meta": {},
                             "jb": {"events": [], "ts": None, "attempt": None}})
        m.load_calendar()
        self.assertEqual([e["title"] for e in m._cal_state["jb"]["events"]],
                         ["Non-Farm Employment Change"])

    def test_credit_exhaustion_is_reported_as_such_not_as_a_bad_key(self):
        msg = m._jb_error_text(401, '{"message":"This endpoint requires credits, '
                                    'and you currently do not have any."}')
        self.assertIn("credit", msg.lower())
        self.assertNotIn("rate", msg.lower())


class Route(unittest.TestCase):
    def setUp(self):
        self._orig = (m.CAL_STATE_FILE, m.fetch_jblanked, m.fetch_ff)
        m.CAL_STATE_FILE = os.path.join(tempfile.mkdtemp(), "calendar_state.json")
        m._cal_state.update({"events": [], "ts": None, "attempt": None, "meta": {},
                             "jb": {"events": [], "ts": None, "attempt": None}})
        os.environ.pop("JBLANKED_API_KEY", None)
        m.fetch_ff = lambda: [FF_ROW]
        m.fetch_jblanked = lambda key: []

    def tearDown(self):
        m.CAL_STATE_FILE, m.fetch_jblanked, m.fetch_ff = self._orig

    def test_calendar_route_returns_events_meta_and_clock(self):
        r = m.app.test_client().get("/calendar")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d["events"][0]["title"], "CPI m/m")
        self.assertEqual(d["meta"]["source"], "forexfactory")
        self.assertTrue(d["ts"])
        self.assertTrue(d["now"].endswith("Z"))

    def test_dashboard_html_has_the_calendar_tab(self):
        html = m.app.test_client().get("/").get_data(as_text=True)
        self.assertIn('data-pg="calendar"', html)
        self.assertIn('id="pg-calendar"', html)
        self.assertIn('id="dash-cal"', html)


if __name__ == "__main__":
    unittest.main()
