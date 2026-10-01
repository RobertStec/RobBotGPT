import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from database import save_memory, search_memory
from rag import retrieve_from_rag

from typing import Any
import requests
import os
from langgraph.types import interrupt
from langgraph.prebuilt import ToolRuntime



load_dotenv()


CURRENT_THREAD_ID = "default"


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


web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get the latest stock quote for a given ticker symbol.

    Args:
        symbol: Stock ticker symbol, for example "AAPL", "TSLA", or "MSFT".

    Returns:
        Latest stock quote information returned by Alpha Vantage.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "status": "error",
            "message": (
                "Alpha Vantage API key is missing. "
                "Set ALPHA_VANTAGE_API_KEY in the .env file."
            )
        }

    symbol = symbol.strip().upper()

    if not symbol:
        return {
            "status": "error",
            "message": "Stock symbol is required."
        }

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
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
        return {
            "status": "error",
            "message": f"Could not connect to Alpha Vantage: {error}"
        }

    except ValueError:
        return {
            "status": "error",
            "message": "Alpha Vantage returned an invalid JSON response."
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

    symbol = symbol.strip().upper()

    if not symbol:
        return {
            "status": "error",
            "message": "Stock symbol is required."
        }

    if quantity <= 0:
        return {
            "status": "error",
            "message": "Quantity must be greater than 0."
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

    api_key = os.getenv("OPENWEATHER_API_KEY")

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
            "appid": api_key,
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
            "appid": api_key,
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
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"




@tool
def search_uploaded_documents(
    query: str,
    runtime: ToolRuntime) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs,
    DOCX, TXT, notes, files, or documents.
    """

    thread_id = get_thread_id(runtime)

    return retrieve_from_rag(
        query=query,
        thread_id=thread_id
    )




@tool
def remember_this(
    memory: str,
    runtime: ToolRuntime) -> str:
    """
    Save an important user preference or fact
    into long-term memory.
    """

    thread_id = get_thread_id(runtime)

    return save_memory(
        thread_id=thread_id,
        memory=memory
    )



@tool
def recall_memory(
    query: str,
    runtime: ToolRuntime) -> str:
    """
    Recall saved long-term memories
    about this conversation.
    """

    thread_id = get_thread_id(runtime)

    return search_memory(
        thread_id=thread_id,
        query=query
    )




tools = [
    calculator,
    get_stock_price,
    purchase_stock,
    get_current_weather,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]