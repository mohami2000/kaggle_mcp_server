"""MCP Server for interacting with Kaggle competition data.

This server provides tools for searching and retrieving information about 
Kaggle competitions using the Meta Kaggle dataset.

Capabilities:
- Search competitions by keywords in title or subtitle.
- Get detailed information for a specific competition by its ID.

Note: This relies on a static snapshot of the Meta Kaggle 'Competitions.csv'. 
It does not interact with the live Kaggle API. Ensure 'Competitions.csv' 
is available locally.
"""

import asyncio
from typing import Optional, List, Dict, Any
import json
import pandas as pd
from pathlib import Path

from mcp.server.models import InitializationOptions 
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl, BaseModel, Field
import mcp.server.stdio


class KaggleCompetitionData:
    """Handles loading and querying Kaggle competition data from CSV."""
    def __init__(self, csv_path: str = "Competitions.csv"):
        self.csv_path = Path(csv_path)
        self.competitions_df: Optional[pd.DataFrame] = None
        self.competitions_by_id: Dict[int, Dict[str, Any]] = {}
        self._load_data()

    def _load_data(self):
        """Loads competition data from the CSV file."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Competitions CSV not found at '{self.csv_path}'. "
                "Download it from kaggle/meta-kaggle dataset."
            )
        try:
            columns_to_use = [
                "Id", "Title", "Subtitle", "HostSegmentTitle", 
                "ForumId", "OrganizationId", "EnabledDate", "DeadlineDate", 
                "RewardType", "RewardQuantity", "UserRankMultiplier", 
                "EvaluationAlgorithmName", "CanQualifyTiers", "TotalTeams", 
                "TotalCompetitors", "TotalSubmissions" 
            ]
            self.competitions_df = pd.read_csv(self.csv_path, usecols=columns_to_use)
            
            self.competitions_df = self.competitions_df.where(pd.notnull(self.competitions_df), None)
            
            self.competitions_by_id = {
                row['Id']: row.to_dict() 
                for _, row in self.competitions_df.iterrows()
            }
            print(f"Loaded {len(self.competitions_by_id)} competitions from {self.csv_path}")
            
        except Exception as e:
            print(f"Error loading competition data from {self.csv_path}: {e}")
            self.competitions_df = pd.DataFrame() 
            self.competitions_by_id = {}

    async def get_competition_info(self, competition_id: int) -> Optional[Dict[str, Any]]:
        """Get details for a specific competition by its ID."""
        return self.competitions_by_id.get(competition_id)

    async def search_competitions(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search competitions by keyword in Title or Subtitle."""
        if self.competitions_df is None or self.competitions_df.empty:
            return []
            
        query = query.lower().strip()
        if not query:
            return []

        title_matches = self.competitions_df[
            self.competitions_df['Title'].str.lower().str.contains(query, na=False)
        ]
        subtitle_matches = self.competitions_df[
            self.competitions_df['Subtitle'].str.lower().str.contains(query, na=False)
        ]
        
        results = pd.concat([title_matches, subtitle_matches]).drop_duplicates(subset=['Id'])
        
        search_results = results.head(max_results)[['Id', 'Title', 'Subtitle']].to_dict('records')
        return search_results


class CompetitionState:
    """Manages competition data access."""
    def __init__(self, csv_path: str = "Competitions.csv"):
        self.data_handler = KaggleCompetitionData(csv_path=csv_path)

    async def get_info(self, competition_id: int) -> Optional[Dict[str, Any]]:
        """Get competition info."""
        return await self.data_handler.get_competition_info(competition_id)

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search competitions."""
        return await self.data_handler.search_competitions(query, max_results)


# --- MCP Server Setup ---
COMPETITION_CSV_PATH = "Competitions.csv" 
server = Server("kaggle-competition-viewer")
try:
    state = CompetitionState(csv_path=COMPETITION_CSV_PATH)
except FileNotFoundError as e:
    print(f"Fatal Error: {e}")
    print("Please ensure 'Competitions.csv' is available.")
    exit(1)


# --- MCP Handlers ---


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for Kaggle competitions."""
    return [
        types.Tool(
            name="get_competition_info",
            description="Get detailed information about a specific Kaggle competition using its numeric ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition_id": {
                        "type": "integer",
                        "description": "The numeric ID of the Kaggle competition (e.g., found via search).",
                        "examples": [3973, 5370]
                    }
                },
                "required": ["competition_id"],
            }
        ),
        types.Tool(
            name="search_competitions",
            description="Search for Kaggle competitions by keywords in their title or subtitle.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for in competition titles or subtitles.",
                        "examples": ["titanic", "house prices", "image classification"]
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return.",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50 
                    }
                },
                "required": ["query"],
            }
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests for Kaggle competitions."""
    if arguments is None:
        arguments = {}

    if name == "get_competition_info":
        if "competition_id" not in arguments:
             return [types.TextContent(type="text", text="Error: Missing 'competition_id' argument.")]
        
        try:
            competition_id = int(arguments["competition_id"])
        except ValueError:
             return [types.TextContent(type="text", text=f"Error: 'competition_id' must be an integer.")]

        info = await state.get_info(competition_id)
        if info:
            serializable_info = {k: (int(v) if isinstance(v, pd.np.integer) else
                                     float(v) if isinstance(v, pd.np.floating) else 
                                     v) 
                                 for k, v in info.items()}
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(serializable_info, indent=2, default=str) 
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text",
                    text=f"Competition with ID {competition_id} not found."
                )
            ]

    elif name == "search_competitions":
        if "query" not in arguments:
             return [types.TextContent(type="text", text="Error: Missing 'query' argument.")]
             
        query = arguments["query"]
        max_results = arguments.get("max_results", 10)
        
        try:
             max_results = int(max_results)
             if not (1 <= max_results <= 50):
                 raise ValueError("max_results must be between 1 and 50")
        except ValueError as e:
             return [types.TextContent(type="text", text=f"Error: Invalid 'max_results'. {e}")]

        results = await state.search(query, max_results)
        if results:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )
            ]
        else:
             return [
                types.TextContent(
                    type="text",
                    text=f"No competitions found matching query: '{query}'"
                )
            ]

    else:
         return [
            types.TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}' requested."
            )
        ]


async def main():
    """Run the server using stdin/stdout streams."""
    print("Starting Kaggle Competition Viewer MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="kaggle-competition-viewer",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
    print("Kaggle Competition Viewer MCP Server stopped.")

if __name__ == "__main__":
    asyncio.run(main())