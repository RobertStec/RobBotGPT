from pydantic import SecretStr
import math

import pytest
import requests
import tools

from types import SimpleNamespace

from tools import (
    calculator,
    get_current_weather,
    get_stock_price,
    get_thread_id,
    normalize_stock_symbol,
    purchase_stock,
    recall_memory,
    remember_this,
    safe_calculate,
)


# Helper tworzący testowy runtime

def make_runtime(thread_id: str | None = "thread-123"):
    configurable = {}

    if thread_id is not None:
        configurable["thread_id"] = thread_id

    return SimpleNamespace(
        config={
            "configurable": configurable
        }
    )



# Pierwsze testy safe_calculate()

@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("10 - 3", 7),
        ("5 * 6", 30),
        ("10 / 4", 2.5),
        ("10 // 3", 3),
        ("10 % 3", 1),
        ("2 ** 8", 256),
    ],
)
def test_safe_calculate_basic_operations(expression, expected):
    assert safe_calculate(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(81)", 9),
        ("math.sqrt(16)", 4),
        ("abs(-10)", 10),
        ("sum([1, 2, 3, 4])", 10),
        ("max(3, 7, 5)", 7),
        ("min(3, 7, 5)", 3),
        ("round(math.pi, 3)", round(math.pi, 3)),
    ],
)
def test_safe_calculate_allowed_functions(expression, expected):
    assert safe_calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "open('file.txt')",
        "__import__('os')",
        "__import__('os').system('dir')",
        "unknown_function(10)",
        "abc",
    ],
)
def test_safe_calculate_rejects_unsafe_expressions(expression):
    with pytest.raises(ValueError):
        safe_calculate(expression)


def test_safe_calculate_rejects_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        safe_calculate("10 / 0")


def test_safe_calculate_rejects_too_large_exponent():
    with pytest.raises(ValueError, match="Exponent is too large"):
        safe_calculate("2 ** 1001")


def test_safe_calculate_rejects_too_long_expression():
    expression = "1+" * 101 + "1"

    with pytest.raises(ValueError, match="Expression is too long"):
        safe_calculate(expression)





@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("AAPL", "AAPL"),
        ("aapl", "AAPL"),
        (" tsla ", "TSLA"),
        ("BRK.B", "BRK.B"),
        ("BTC-USD", "BTC-USD"),
    ],
)
def test_normalize_stock_symbol_valid(symbol, expected):
    assert normalize_stock_symbol(symbol) == expected


@pytest.mark.parametrize(
    "symbol",
    [
        "",
        "   ",
        "AAPL!",
        "ABC@123",
        "THIS_SYMBOL_IS_WAY_TOO_LONG",
    ],
)
def test_normalize_stock_symbol_invalid(symbol):
    with pytest.raises(ValueError):
        normalize_stock_symbol(symbol)




# Testujemy właściwy tool calculator

def test_calculator_tool_returns_result_as_string():
    result = calculator.invoke({
        "expression": "125 * 48 / 6"
    })

    assert result == "1000.0"


def test_calculator_tool_handles_invalid_expression():
    result = calculator.invoke({
        "expression": "10 / 0"
    })

    assert result.startswith("Calculation error:")
    assert "division by zero" in result.lower()


def test_calculator_tool_rejects_unsafe_expression():
    result = calculator.invoke({
        "expression": "__import__('os').system('dir')"
    })

    assert result.startswith("Calculation error:")



# get_stock_price — brak API key

def test_get_stock_price_returns_error_when_api_key_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        None,
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL"
    })

    assert result["status"] == "error"
    assert "API key is missing" in result["message"]



# Mockujemy poprawną odpowiedź Alpha Vantage


def test_get_stock_price_returns_stock_data(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mock_response = mocker.Mock()

    mock_response.raise_for_status.return_value = None

    mock_response.json.return_value = {
        "Global Quote": {
            "01. symbol": "AAPL",
            "05. price": "250.00",
            "09. change": "2.50",
            "10. change percent": "1.01%",
            "08. previous close": "247.50",
            "07. latest trading day": "2026-10-01",
        }
    }

    mock_get = mocker.patch(
        "tools.requests.get",
        return_value=mock_response,
    )

    result = get_stock_price.invoke({
        "symbol": "aapl"
    })

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"
    assert result["price"] == "250.00"
    assert result["previous_close"] == "247.50"

    mock_get.assert_called_once()


# Test timeoutu Alpha Vantage

def test_get_stock_price_handles_timeout(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mocker.patch(
        "tools.requests.get",
        side_effect=requests.Timeout,
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL"
    })

    assert result["status"] == "error"
    assert "timed out" in result["message"].lower()



# Testujemy get_current_weather
# Najpierw brak API key

def test_get_current_weather_returns_error_when_api_key_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        tools.settings,
        "openweather_api_key",
        None,
    )

    result = get_current_weather.invoke({
        "location": "Wroclaw"
    })

    assert "API key is missing" in result



# Mockujemy oba requesty pogodowe

def test_get_current_weather_returns_weather_data(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "openweather_api_key",
        SecretStr("test-key"),
    )

    geo_response = mocker.Mock()

    geo_response.raise_for_status.return_value = None

    geo_response.json.return_value = [
        {
            "name": "Wrocław",
            "lat": 51.1079,
            "lon": 17.0385,
            "country": "PL",
            "state": "Lower Silesia",
        }
    ]

    weather_response = mocker.Mock()

    weather_response.raise_for_status.return_value = None

    weather_response.json.return_value = {
        "main": {
            "temp": 18.5,
            "feels_like": 18.0,
            "humidity": 65,
            "pressure": 1015,
        },
        "weather": [
            {
                "description": "clear sky"
            }
        ],
        "wind": {
            "speed": 3.2
        },
        "visibility": 10000,
    }

    mock_get = mocker.patch(
        "tools.requests.get",
        side_effect=[
            geo_response,
            weather_response,
        ],
    )

    result = get_current_weather.invoke({
        "location": "Wroclaw"
    })

    assert "Wrocław" in result
    assert "18.5°C" in result
    assert "Clear Sky" in result
    assert "65%" in result
    assert "1015 hPa" in result
    assert "3.2 m/s" in result
    assert "10.0 km" in result

    assert mock_get.call_count == 2


# Test sytuacji „miasto nie istnieje”

def test_get_current_weather_handles_unknown_location(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "openweather_api_key",
        SecretStr("test-key"),
    )

    geo_response = mocker.Mock()

    geo_response.raise_for_status.return_value = None
    geo_response.json.return_value = []

    mock_get = mocker.patch(
        "tools.requests.get",
        return_value=geo_response,
    )

    result = get_current_weather.invoke({
        "location": "DefinitelyNotARealCity123"
    })

    assert "Could not find the location" in result

    mock_get.assert_called_once()



# Test timeoutu OpenWeather

def test_get_current_weather_handles_timeout(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "openweather_api_key",
        SecretStr("test-key"),
    )

    mocker.patch(
        "tools.requests.get",
        side_effect=requests.Timeout,
    )

    result = get_current_weather.invoke({
        "location": "Wroclaw"
    })

    assert "timed out" in result.lower()



# Testujemy get_thread_id()

def test_get_thread_id_returns_thread_id():
    runtime = make_runtime(
        thread_id="test-thread-001"
    )

    result = get_thread_id(runtime)

    assert result == "test-thread-001"


def test_get_thread_id_raises_error_when_thread_id_missing():
    runtime = make_runtime(
        thread_id=None
    )

    with pytest.raises(
        ValueError,
        match="thread_id is missing",
    ):
        get_thread_id(runtime)



# Testujemy remember_this

def test_remember_this_saves_memory(
    mocker,
):
    runtime = make_runtime(
        thread_id="thread-memory-001"
    )

    mock_save_memory = mocker.patch(
        "tools.save_memory",
        return_value="Memory saved successfully.",
    )

    result = remember_this.func(
        memory="  I prefer Python.  ",   # oczekujemy "I prefer Python."
        runtime=runtime,
    )

    assert result == "Memory saved successfully."

    mock_save_memory.assert_called_once_with(
        thread_id="thread-memory-001",
        memory="I prefer Python.",
    )


# Pusta pamięć nie powinna trafić do DB

def test_remember_this_rejects_empty_memory(
    mocker,
):
    runtime = make_runtime()

    mock_save_memory = mocker.patch(
        "tools.save_memory"
    )

    result = remember_this.func(
        memory="   ",
        runtime=runtime,
    )

    assert result == "Nothing to remember."

    mock_save_memory.assert_not_called()


# Za długa pamięć

def test_remember_this_rejects_too_long_memory(
    mocker,
):
    runtime = make_runtime()

    mock_save_memory = mocker.patch(
        "tools.save_memory"
    )

    result = remember_this.func(
        memory="x" * 2001,
        runtime=runtime,
    )

    assert "Memory is too long" in result

    mock_save_memory.assert_not_called()



# Test recall_memory

def test_recall_memory_searches_memory(
mocker,
):
    runtime = make_runtime(
        thread_id="thread-memory-002"
    )

    mock_search_memory = mocker.patch(
        "tools.search_memory",
        return_value="- I prefer Python.",
    )

    result = recall_memory.func(
        query="  programming preferences  ",
        runtime=runtime,
    )

    assert result == "- I prefer Python."

    mock_search_memory.assert_called_once_with(
        thread_id="thread-memory-002",
        query="programming preferences",
    )


# Puste zapytanie pamięci

def test_recall_memory_rejects_empty_query(
    mocker,
):
    runtime = make_runtime()

    mock_search_memory = mocker.patch(
        "tools.search_memory"
    )

    result = recall_memory.func(
        query="   ",
        runtime=runtime,
    )

    assert result == "Memory query is required."

    mock_search_memory.assert_not_called()



# Za długie zapytanie. Limit wynosi 500 znaków.

def test_recall_memory_rejects_too_long_query(
    mocker,
):
    runtime = make_runtime()

    mock_search_memory = mocker.patch(
        "tools.search_memory"
    )

    result = recall_memory.func(
        query="x" * 501,
        runtime=runtime,
    )

    assert result == "Memory query is too long."

    mock_search_memory.assert_not_called()



# HITL purchase_stock
# Test zatwierdzonego zakupu

def test_purchase_stock_returns_success_when_approved(
    mocker,
):
    mock_interrupt = mocker.patch(
        "tools.interrupt",
        return_value=True,
    )

    result = purchase_stock.invoke({
        "symbol": "aapl",
        "quantity": 5,
    })

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"
    assert result["quantity"] == 5

    mock_interrupt.assert_called_once_with({
        "type": "stock_purchase_approval",
        "action": "purchase_stock",
        "symbol": "AAPL",
        "quantity": 5,
        "message": (
            "Do you approve purchasing "
            "5 shares of AAPL?"
        ),
    })


# Test odrzucenia zakupu

def test_purchase_stock_returns_cancelled_when_declined(
    mocker,
):
    mocker.patch(
        "tools.interrupt",
        return_value=False,
    )

    result = purchase_stock.invoke({
        "symbol": "TSLA",
        "quantity": 3,
    })

    assert result["status"] == "cancelled"
    assert result["symbol"] == "TSLA"
    assert result["quantity"] == 3

    assert "declined" in result["message"].lower()



# Niepoprawna liczba akcji - interrupt() nie powinien zostać wywołany, jeśli request jest niepoprawny.

@pytest.mark.parametrize(
    "quantity",
    [
        0,
        -1,
        -100,
    ],
)
def test_purchase_stock_rejects_non_positive_quantity(
    quantity,
    mocker,
):
    mock_interrupt = mocker.patch(
        "tools.interrupt"
    )

    result = purchase_stock.invoke({
        "symbol": "AAPL",
        "quantity": quantity,
    })

    assert result["status"] == "error"
    assert "greater than 0" in result["message"]

    mock_interrupt.assert_not_called()



# Zbyt duży zakup

def test_purchase_stock_rejects_too_large_quantity(
    mocker,
):
    mock_interrupt = mocker.patch(
        "tools.interrupt"
    )

    result = purchase_stock.invoke({
        "symbol": "AAPL",
        "quantity": 100_001,
    })

    assert result["status"] == "error"
    assert "too large" in result["message"].lower()

    mock_interrupt.assert_not_called()


# Niepoprawny symbol również nie może uruchomić HITL

def test_purchase_stock_rejects_invalid_symbol_before_interrupt(
    mocker,
):
    mock_interrupt = mocker.patch(
        "tools.interrupt"
    )

    result = purchase_stock.invoke({
        "symbol": "AAPL!!!",
        "quantity": 10,
    })

    assert result["status"] == "error"
    assert "Invalid stock symbol" in result["message"]

    mock_interrupt.assert_not_called()



# =========================================================
# Additional stock API coverage
# =========================================================


def test_get_stock_price_rejects_invalid_symbol(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mock_get = mocker.patch(
        "tools.requests.get"
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL!!!"
    })

    assert result["status"] == "error"
    assert "Invalid stock symbol" in result["message"]

    mock_get.assert_not_called()


def test_get_stock_price_handles_alpha_vantage_error_message(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None

    mock_response.json.return_value = {
        "Error Message": "Invalid API call."
    }

    mocker.patch(
        "tools.requests.get",
        return_value=mock_response,
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL"
    })

    assert result == {
        "status": "error",
        "message": "Invalid stock symbol: AAPL",
    }


def test_get_stock_price_handles_empty_global_quote(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None

    mock_response.json.return_value = {
        "Global Quote": {}
    }

    mocker.patch(
        "tools.requests.get",
        return_value=mock_response,
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL"
    })

    assert result["status"] == "error"

    assert (
        result["message"]
        == "No stock data found for symbol: AAPL"
    )


def test_get_stock_price_handles_http_error(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mock_response = mocker.Mock()
    mock_response.status_code = 503

    mock_response.raise_for_status.side_effect = (
        requests.HTTPError(
            response=mock_response
        )
    )

    mocker.patch(
        "tools.requests.get",
        return_value=mock_response,
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL"
    })

    assert result == {
        "status": "error",
        "message": (
            "Alpha Vantage returned HTTP error: 503"
        ),
    }


def test_get_stock_price_handles_connection_error(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mocker.patch(
        "tools.requests.get",
        side_effect=requests.ConnectionError(
            "connection failed"
        ),
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL"
    })

    assert result == {
        "status": "error",
        "message": (
            "Could not connect to "
            "the stock price service."
        ),
    }


def test_get_stock_price_handles_invalid_json(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "alpha_vantage_api_key",
        SecretStr("test-key"),
    )

    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None

    mock_response.json.side_effect = ValueError(
        "invalid json"
    )

    mocker.patch(
        "tools.requests.get",
        return_value=mock_response,
    )

    result = get_stock_price.invoke({
        "symbol": "AAPL"
    })

    assert result == {
        "status": "error",
        "message": (
            "Alpha Vantage returned "
            "an invalid JSON response."
        ),
    }



# =========================================================
# Additional weather coverage
# =========================================================


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (
            "   ",
            "Weather location is required.",
        ),
        (
            "x" * 121,
            "Weather location is too long.",
        ),
    ],
)
def test_get_current_weather_validates_location(
    location,
    expected,
    mocker,
):
    mock_get = mocker.patch(
        "tools.requests.get"
    )

    result = get_current_weather.invoke({
        "location": location
    })

    assert result == expected

    mock_get.assert_not_called()


def test_get_current_weather_handles_401(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "openweather_api_key",
        SecretStr("test-key"),
    )

    mock_response = mocker.Mock()
    mock_response.status_code = 401

    mock_response.raise_for_status.side_effect = (
        requests.HTTPError(
            response=mock_response
        )
    )

    mocker.patch(
        "tools.requests.get",
        return_value=mock_response,
    )

    result = get_current_weather.invoke({
        "location": "Wroclaw"
    })

    assert result == (
        "The OpenWeather API key "
        "is invalid or inactive."
    )


def test_get_current_weather_handles_connection_error(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "openweather_api_key",
        SecretStr("test-key"),
    )

    mocker.patch(
        "tools.requests.get",
        side_effect=requests.ConnectionError(
            "connection failed"
        ),
    )

    result = get_current_weather.invoke({
        "location": "Wroclaw"
    })

    assert result == (
        "Could not connect to "
        "the weather service."
    )


def test_get_current_weather_handles_unexpected_response(
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        tools.settings,
        "openweather_api_key",
        SecretStr("test-key"),
    )

    geo_response = mocker.Mock()
    geo_response.raise_for_status.return_value = None

    geo_response.json.return_value = [
        {
            "name": "Wrocław",
            "lat": 51.1079,
            "lon": 17.0385,
            "country": "PL",
        }
    ]

    weather_response = mocker.Mock()
    weather_response.raise_for_status.return_value = None

    # Missing "main" and "weather".
    weather_response.json.return_value = {}

    mocker.patch(
        "tools.requests.get",
        side_effect=[
            geo_response,
            weather_response,
        ],
    )

    result = get_current_weather.invoke({
        "location": "Wroclaw"
    })

    assert result == (
        "The weather service returned "
        "an unexpected response."
    )




# =========================================================
# Memory error coverage
# =========================================================


def test_remember_this_handles_database_error(
    mocker,
):
    runtime = make_runtime(
        "thread-memory-error"
    )

    mocker.patch(
        "tools.save_memory",
        side_effect=RuntimeError(
            "database unavailable"
        ),
    )

    result = remember_this.func(
        memory="Remember this",
        runtime=runtime,
    )

    assert result == (
        "The memory tool could not "
        "save the information."
    )


def test_recall_memory_handles_database_error(
    mocker,
):
    runtime = make_runtime(
        "thread-memory-error"
    )

    mocker.patch(
        "tools.search_memory",
        side_effect=RuntimeError(
            "database unavailable"
        ),
    )

    result = recall_memory.func(
        query="preferences",
        runtime=runtime,
    )

    assert result == (
        "The memory tool could not "
        "retrieve the information."
    )
