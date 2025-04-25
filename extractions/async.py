import asyncio
import dataclasses
from typing import Callable, List, Optional

import httpx

MAX_RETRIES = 3
RETRY_DELAY = 2.0


@dataclasses.dataclass
class Credentials:
    """To store username and password."""
    username: str
    password: str


async def authenticate_api(
    client: httpx.AsyncClient,
    auth_url: str,
    payload: dict,
    headers: dict
) -> Optional[dict]:
    """
    Authenticate with an API server and retrieve a bearer token.

    Returns:
        Dictionary with Authorization header or None if authentication fails.
    """
    try:
        response = await client.post(auth_url, data=payload, headers=headers)
        response.raise_for_status()
        token = response.json()["access_token"]
        print(f"Authentication successful for {auth_url}")
        return {"Authorization": f"Bearer {token}"}
    except httpx.HTTPError as e:
        print(f"Authentication failed for {auth_url}: {e}")
        return None


async def get_authorisations(tasks: List[Callable]) -> List[Optional[dict]]:
    """
    Authenticate with multiple API servers.

    Args:
        tasks: List of async functions that each take an httpx.AsyncClient.

    Returns:
        List of results from each task.
    """
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[task(client) for task in tasks])


async def get_data(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    timeout: float = 5.0
) -> Optional[dict]:
    """Perform an async GET request with retries."""
    print(f"Sending GET request to {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            print(f"GET request succeeded for {url}")
            return response.json()
        except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            print(f"Timeout error on attempt {attempt} for {url}: {e}")
        except httpx.HTTPError as e:
            print(f"HTTP error on attempt {attempt} for {url}: {e}")

        if attempt < MAX_RETRIES:
            print("Retrying...\n")
            await asyncio.sleep(RETRY_DELAY)
        else:
            print(f"Max retries ({MAX_RETRIES}) exceeded for {url}\n")
            return None


async def post_data(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: Optional[dict] = None,
    timeout: float = 5.0
) -> Optional[dict]:
    """Perform an async POST request with retries."""
    print(f"Sending POST request to {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            print(f"POST request succeeded for {url}")
            return response.json()
        except (httpx.ReadTimeout, httpx.WriteTimeout) as e:
            print(f"Timeout on POST attempt {attempt} to {url}: {e}")
        except httpx.HTTPError as e:
            print(f"HTTP error on POST attempt {attempt} to {url}: {e}")

        if attempt < MAX_RETRIES:
            print("Retrying...\n")
            await asyncio.sleep(RETRY_DELAY)
        else:
            print(f"Max retries ({MAX_RETRIES}) exceeded for {url}\n")
            return None


async def put_data(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: Optional[dict] = None,
    timeout: float = 5.0
) -> Optional[dict]:
    """Perform an async PUT request with retries."""
    print(f"Sending PUT request to {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.put(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            print(f"PUT request succeeded for {url}")
            return response.json()
        except httpx.HTTPError as e:
            print(f"HTTP error on PUT attempt {attempt} to {url}: {e}")

        if attempt < MAX_RETRIES:
            print("Retrying...\n")
            await asyncio.sleep(RETRY_DELAY)
        else:
            print(f"Max retries ({MAX_RETRIES}) exceeded for {url}\n")
            return None


async def fetch_with_semaphore(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    query_url: str,
    headers: dict,
    statement: str = '',
    timeout: float = 5.0
) -> Optional[dict]:
    """Helper function to call get_data with a semaphore."""
    async with semaphore:
        print(statement)
        return await get_data(client, query_url, headers, timeout)


async def post_with_semaphore(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    query_url: str,
    headers: dict,
    payload: dict,
    statement: str = '',
    timeout: float = 5.0
) -> Optional[dict]:
    """Helper function to call post_data with a semaphore."""
    async with semaphore:
        print(statement)
        return await post_data(client, query_url, headers, payload, timeout)
