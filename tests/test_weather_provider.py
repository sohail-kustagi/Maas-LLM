"""Tests for WeatherProvider — fake provider, Open-Meteo API shape validation."""
import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

from src.core.weather_provider import WeatherProvider
from src.core.weather_types import WeatherSnapshot


class FakeWeatherProviderTests(unittest.TestCase):
    """Test the fake (deterministic) provider which requires no network."""

    def setUp(self):
        self.provider = WeatherProvider(use_fake=True)

    def _get(self, lat=0.0, lon=0.0) -> WeatherSnapshot:
        return asyncio.run(self.provider.get_weather(lat, lon))

    def test_fake_returns_snapshot(self):
        snap = self._get()
        self.assertIsInstance(snap, WeatherSnapshot)

    def test_fake_has_expected_safe_values(self):
        snap = self._get()
        self.assertEqual(snap.source, "fake")
        self.assertEqual(snap.confidence, 1.0)
        self.assertEqual(snap.lightning_risk, 0.0)
        self.assertEqual(snap.thunderstorm_risk, 0.0)
        self.assertGreater(snap.visibility_m, 1000.0)

    def test_fake_returns_correct_location(self):
        snap = self._get(lat=12.34, lon=56.78)
        self.assertAlmostEqual(snap.latitude, 12.34)
        self.assertAlmostEqual(snap.longitude, 56.78)

    def test_fake_timestamp_is_recent(self):
        before = time.time()
        snap = self._get()
        after = time.time()
        self.assertGreaterEqual(snap.timestamp, before)
        self.assertLessEqual(snap.timestamp, after)

    def test_fake_wind_values_in_safe_range(self):
        snap = self._get()
        self.assertLess(snap.wind_speed_mps, 15.0)
        self.assertLess(snap.gust_speed_mps, 20.0)


class OpenMeteoProviderTests(unittest.TestCase):
    """Test the real Open-Meteo provider using a mocked HTTP response."""

    def _make_mock_response(self, current_data: dict) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"current": current_data}
        return mock_resp

    def _get_real(self, lat=0.0, lon=0.0, current_data=None) -> WeatherSnapshot:
        if current_data is None:
            current_data = {
                "temperature_2m": 20.0,
                "wind_speed_10m": 18.0,   # km/h → 5 m/s
                "wind_direction_10m": 270.0,
                "wind_gusts_10m": 25.0,   # km/h → ~6.9 m/s
                "precipitation": 1.0,
                "weather_code": 3,        # overcast, no storm
            }
        provider = WeatherProvider(use_fake=False)
        with patch("src.core.weather_provider.requests.get", return_value=self._make_mock_response(current_data)):
            return asyncio.run(provider.get_weather(lat, lon))

    def test_real_provider_returns_snapshot(self):
        snap = self._get_real()
        self.assertIsInstance(snap, WeatherSnapshot)

    def test_wind_conversion_km_h_to_m_s(self):
        # 18 km/h → 5.0 m/s
        snap = self._get_real(current_data={
            "wind_speed_10m": 18.0,
            "wind_direction_10m": 0.0,
            "wind_gusts_10m": 0.0,
            "precipitation": 0.0,
            "temperature_2m": 20.0,
            "weather_code": 0,
        })
        self.assertAlmostEqual(snap.wind_speed_mps, 18.0 / 3.6, places=2)

    def test_thunderstorm_weather_code_sets_risk(self):
        snap = self._get_real(current_data={
            "wind_speed_10m": 0.0, "wind_direction_10m": 0.0,
            "wind_gusts_10m": 0.0, "precipitation": 0.0,
            "temperature_2m": 18.0,
            "weather_code": 95,  # WMO thunderstorm code
        })
        self.assertEqual(snap.thunderstorm_risk, 1.0)
        self.assertEqual(snap.lightning_risk, 1.0)

    def test_clear_weather_code_has_zero_risk(self):
        snap = self._get_real(current_data={
            "wind_speed_10m": 0.0, "wind_direction_10m": 0.0,
            "wind_gusts_10m": 0.0, "precipitation": 0.0,
            "temperature_2m": 25.0,
            "weather_code": 0,  # Clear sky
        })
        self.assertEqual(snap.thunderstorm_risk, 0.0)
        self.assertEqual(snap.lightning_risk, 0.0)

    def test_source_is_open_meteo(self):
        snap = self._get_real()
        self.assertEqual(snap.source, "open-meteo")

    def test_network_error_returns_none(self):
        """If requests raises, provider must return None (not propagate)."""
        provider = WeatherProvider(use_fake=False)
        with patch("src.core.weather_provider.requests.get", side_effect=Exception("timeout")):
            result = asyncio.run(provider.get_weather(0.0, 0.0))
        self.assertIsNone(result)

    def test_correct_params_sent_to_api(self):
        """Ensure the API call is made with the correct parameters."""
        provider = WeatherProvider(use_fake=False)
        with patch("src.core.weather_provider.requests.get", return_value=self._make_mock_response({
            "wind_speed_10m": 0.0, "wind_direction_10m": 0.0,
            "wind_gusts_10m": 0.0, "precipitation": 0.0,
            "temperature_2m": 20.0, "weather_code": 0,
        })) as mock_get:
            asyncio.run(provider.get_weather(lat=-35.36, lon=149.16))
            call_kwargs = mock_get.call_args
            params = call_kwargs[1].get("params") or call_kwargs[0][1]
            self.assertEqual(params["latitude"], -35.36)
            self.assertEqual(params["longitude"], 149.16)
            self.assertIn("current", params)
            self.assertIsInstance(params["current"], str)  # must be comma-separated string


if __name__ == "__main__":
    unittest.main()
