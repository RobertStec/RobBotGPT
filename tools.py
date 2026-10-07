import logging
import ast
import operator
import re
import math

from langchain_core.tools import tool
from core.config import settings
from langchain_tavily import TavilySearch

from database import save_memory, search_memory

from typing import Any
import requests
import os
from langgraph.types import interrupt
from langgraph.prebuilt import ToolRuntime




logger = logging.getLogger(__name__)


def get_thread_id(runtime: ToolRuntime) -> str:
    """
    Get the current LangGraph thread_id
    from the tool runtime configuration.
    """

    thread_id = (
        runtime.config
        .get("configurable", {})
        .get("thread_id")
    )

    if not thread_id:
        raise ValueError(
            "thread_id is missing from LangGraph runtime config."
        )

    return thread_id

if settings.tavily_api_key is not None:
    os.environ.setdefault(
        "TAVILY_API_KEY",
        settings.tavily_api_key.get_secret_value(),
    )

web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,

    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_MATH_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

def _resolve_math_function(node):
    """
    Return an allowed math function.
    """

    if isinstance(node, ast.Name):

        function = _ALLOWED_MATH_FUNCTIONS.get(
            node.id
        )

        if function is None:
            raise ValueError(
                f"Function '{node.id}' is not allowed."
            )

        return function

    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "math"
    ):

        function = _ALLOWED_MATH_FUNCTIONS.get(
            node.attr
        )

        if function is None:
            raise ValueError(
                f"Function 'math.{node.attr}' is not allowed."
            )

        return function

    raise ValueError(
        "Only approved math functions are allowed."
    )


def _safe_eval_node(node):

    if isinstance(node, ast.Expression):
        return _safe_eval_node(
            node.body
        )

    # -----------------------------------------
    # Numbers
    # -----------------------------------------

    if isinstance(node, ast.Constant):

        if type(node.value) not in (
            int,
            float,
        ):
            raise ValueError(
                "Only numeric constants are allowed."
            )

        return node.value

    # -----------------------------------------
    # Binary operations
    # -----------------------------------------

    if isinstance(node, ast.BinOp):

        operator_type = type(node.op)

        operation = _BINARY_OPERATORS.get(
            operator_type
        )

        if operation is None:
            raise ValueError(
                "Operator is not allowed."
            )

        left = _safe_eval_node(
            node.left
        )

        right = _safe_eval_node(
            node.right
        )

        if (
            type(left) not in (int, float)
            or type(right) not in (int, float)
        ):
            raise ValueError(
                "Binary operations are allowed "
                "only on numbers."
            )

        # Prevent extremely expensive powers.
        if (
            operator_type is ast.Pow
            and abs(right) > 1000
        ):
            raise ValueError(
                "Exponent is too large."
            )

        return operation(
            left,
            right
        )

    # -----------------------------------------
    # Unary operations
    # -----------------------------------------

    if isinstance(node, ast.UnaryOp):

        operation = _UNARY_OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise ValueError(
                "Unary operator is not allowed."
            )

        return operation(
            _safe_eval_node(
                node.operand
            )
        )

    # -----------------------------------------
    # Constants: pi, e
    # -----------------------------------------

    if isinstance(node, ast.Name):

        if node.id in _ALLOWED_MATH_CONSTANTS:
            return _ALLOWED_MATH_CONSTANTS[
                node.id
            ]

        raise ValueError(
            f"Name '{node.id}' is not allowed."
        )

    # math.pi / math.e
    if isinstance(node, ast.Attribute):

        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "math"
            and node.attr
            in _ALLOWED_MATH_CONSTANTS
        ):
            return _ALLOWED_MATH_CONSTANTS[
                node.attr
            ]

        raise ValueError(
            "Attribute access is not allowed."
        )

    # -----------------------------------------
    # Lists / tuples
    # Allows: sum([1, 2, 3])
    # -----------------------------------------

    if isinstance(node, ast.List):

        return [
            _safe_eval_node(item)
            for item in node.elts
        ]

    if isinstance(node, ast.Tuple):

        return tuple(
            _safe_eval_node(item)
            for item in node.elts
        )

    # -----------------------------------------
    # Function calls
    # -----------------------------------------

    if isinstance(node, ast.Call):

        if node.keywords:
            raise ValueError(
                "Keyword arguments are not allowed."
            )

        function = _resolve_math_function(
            node.func
        )

        arguments = [
            _safe_eval_node(argument)
            for argument in node.args
        ]

        return function(
            *arguments
        )

    raise ValueError(
        f"Unsupported expression: "
        f"{type(node).__name__}"
    )

def safe_calculate(
    expression: str
):
    """
    Parse and calculate a math expression
    without using eval().
    """

    expression = expression.strip()

    if not expression:
        raise ValueError(
            "Expression is required."
        )

    # Avoid extremely large input.
    if len(expression) > 200:
        raise ValueError(
            "Expression is too long."
        )

    tree = ast.parse(
        expression,
        mode="eval"
    )

    # Prevent deliberately huge AST trees.
    node_count = sum(
        1
        for _ in ast.walk(tree)
    )

    if node_count > 100:
        raise ValueError(
            "Expression is too complex."
        )

    result = _safe_eval_node(
        tree
    )

    if (
        isinstance(result, float)
        and not math.isfinite(result)
    ):
        raise ValueError(
            "Result is not finite."
        )

    if (
        isinstance(result, int)
        and len(str(abs(result))) > 1000
    ):
        raise ValueError(
            "Result is too large."
        )

    return result


@tool
def calculator(expression: str) -> str:
    """
    Safely calculate a mathematical expression.

    Examples:
    - 2 + 2
    - 125 * 48 / 6
    - math.sqrt(16)
    - sqrt(81)
    - round(math.pi, 3)
    - sum([1, 2, 3])
    """

    try:

        result = safe_calculate(
            expression
        )

        return str(result)

    except SyntaxError:
        return (
            "Calculation error: "
            "invalid mathematical expression."
        )

    except (
        ValueError,
        TypeError,
        ZeroDivisionError,
        OverflowError,
    ) as error:

        return (
            "Calculation error: "
            f"{error}"
        )

    except Exception:

        logger.exception(
            "Unexpected error in calculator"
        )

        return (
            "The calculator could not complete "
            "the request."
        )




def normalize_stock_symbol(
    symbol: str
) -> str:

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError(
            "Stock symbol is required."
        )

    if not re.fullmatch(
        r"[A-Z0-9.-]{1,15}",
        symbol
    ):
        raise ValueError(
            "Invalid stock symbol."
        )

    return symbol
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get the latest stock quote for a given ticker symbol.

    Args:
        symbol: Stock ticker symbol, for example "AAPL", "TSLA", or "MSFT".

    Returns:
        Latest stock quote information returned by Alpha Vantage.
    """

    api_key = (
        settings.alpha_vantage_api_key
    )

    if api_key is None:
        return {
            "status": "error",
            "message": (
                "Alpha Vantage API key "
                "is missing."
            ),
        }

    api_key_value = (
        api_key.get_secret_value()
    )

    if not api_key:
        return {
            "status": "error",
            "message": (
                "Alpha Vantage API key is missing. "
                "Set ALPHA_VANTAGE_API_KEY in the .env file."
            )
        }

    try:
        symbol = normalize_stock_symbol(
            symbol
        )

    except ValueError as error:
        return {
            "status": "error",
            "message": str(error)
        }

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key_value,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        # Alpha Vantage may return API/rate-limit information
        if "Error Message" in data:
            return {
                "status": "error",
                "message": f"Invalid stock symbol: {symbol}"
            }

        if "Note" in data:
            return {
                "status": "error",
                "message": data["Note"]
            }

        if "Information" in data:
            return {
                "status": "error",
                "message": data["Information"]
            }

        quote = data.get("Global Quote", {})

        if not quote:
            return {
                "status": "error",
                "message": f"No stock data found for symbol: {symbol}"
            }

        return {
            "status": "success",
            "symbol": quote.get("01. symbol", symbol),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "previous_close": quote.get("08. previous close"),
            "latest_trading_day": quote.get("07. latest trading day")
        }

    except requests.Timeout:
        return {
            "status": "error",
            "message": "Alpha Vantage request timed out."
        }

    except requests.HTTPError as error:
        return {
            "status": "error",
            "message": (
                f"Alpha Vantage returned HTTP error: "
                f"{error.response.status_code}"
            )
        }

    except requests.RequestException as error:

        logger.warning(
            "Stock price service request failed: %s",
            error,
        )

        return {
            "status": "error",
            "message": (
                "Could not connect to "
                "the stock price service."
            ),
        }

    except ValueError:
        return {
            "status": "error",
            "message": "Alpha Vantage returned an invalid JSON response."
        }

    except Exception:

        logger.exception(
            "Unexpected error in get_stock_price"
        )

        return {
            "status": "error",
            "message": (
                "The stock price tool could not "
                "complete the request."
            ),
        }


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing shares of a stock.

    HUMAN-IN-THE-LOOP:
    The purchase requires explicit human approval before it can continue.

    Args:
        symbol: Stock ticker symbol, for example "AAPL" or "TSLA".
        quantity: Number of shares to purchase.
    """

    try:
        symbol = normalize_stock_symbol(
            symbol
        )

    except ValueError as error:
        return {
            "status": "error",
            "message": str(error)
        }

    if quantity <= 0:
        return {
            "status": "error",
            "message": "Quantity must be greater than 0."
        }

    if quantity > 100_000:
        return {
            "status": "error",
            "message": (
                "Quantity is too large."
            )
        }

    approval = interrupt({
        "type": "stock_purchase_approval",
        "action": "purchase_stock",
        "symbol": symbol,
        "quantity": quantity,
        "message": (
            f"Do you approve purchasing "
            f"{quantity} shares of {symbol}?"
        )
    })

    if approval is True:
        return {
            "status": "success",
            "message": (
                f"Purchase order placed for "
                f"{quantity} shares of {symbol}."
            ),
            "symbol": symbol,
            "quantity": quantity
        }

    return {
        "status": "cancelled",
        "message": (
            f"Purchase of {quantity} shares "
            f"of {symbol} was declined."
        ),
        "symbol": symbol,
        "quantity": quantity
    }



@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    location = location.strip()

    if not location:
        return (
            "Weather location is required."
        )

    if len(location) > 120:
        return (
            "Weather location is too long."
        )


    api_key = (
        settings.openweather_api_key
    )

    if api_key is None:
        return (
            "Weather API key is missing."
        )

    api_key_value = (
        api_key.get_secret_value()
    )

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key_value,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key_value,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:

        logger.warning(
            "Weather service request failed: %s",
            error,
        )

        return (
            "Could not connect to "
            "the weather service."
        )

    except (KeyError, TypeError, ValueError) as error:

        logger.warning(
            "Unexpected weather API response: %s",
            error,
        )

        return (
            "The weather service returned "
            "an unexpected response."
        )

    except Exception:

        logger.exception(
            "Unexpected error in get_current_weather"
        )

        return (
            "The weather tool could not complete "
            "the request."
        )




@tool
def remember_this(
    memory: str,
    runtime: ToolRuntime
) -> str:
    """
    Save an important user preference or fact
    into long-term memory.
    """

    try:

        thread_id = get_thread_id(
            runtime
        )

        memory = memory.strip()

        if not memory:
            return "Nothing to remember."

        if len(memory) > 2000:
            return (
                "Memory is too long. "
                "Please provide a shorter fact."
            )

        return save_memory(
            thread_id=thread_id,
            memory=memory
        )

    except Exception:

        logger.exception(
            "Failed to save long-term memory"
        )

        return (
            "The memory tool could not "
            "save the information."
        )



@tool
def recall_memory(
    query: str,
    runtime: ToolRuntime
) -> str:
    """
    Recall saved long-term memories
    about this conversation.
    """

    try:

        thread_id = get_thread_id(
            runtime
        )

        query = query.strip()

        if not query:
            return (
                "Memory query is required."
            )

        if len(query) > 500:
            return (
                "Memory query is too long."
            )

        return search_memory(
            thread_id=thread_id,
            query=query
        )

    except Exception:

        logger.exception(
            "Failed to retrieve long-term memory"
        )

        return (
            "The memory tool could not "
            "retrieve the information."
        )




tools = [
    calculator,
    get_stock_price,
    purchase_stock,
    get_current_weather,
    remember_this,
    recall_memory,
    web_search
]