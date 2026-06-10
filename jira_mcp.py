import os
import asyncio
import re
import os
import base64
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

async def run_atlassian_tool():
    # 1. Retrieve secrets securely from Colab environment
    email = os.environ.get('ATLASSIAN_USER_EMAIL')
    token = os.environ.get('ATLASSIAN_API_TOKEN')
    print (f"email is: {email}")
    print (f"token is: {token}")
  
    # 2. Atlassian non-interactive clients require Base64 Basic Auth Header
    auth_str = f"{email}:{token}"
    encoded_auth = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/json"
    }
    
    # 3. Target official remote Atlassian Rovo MCP gateway endpoint
    server_url = "https://mcp.atlassian.com/v1/mcp"
  
    print("Connecting to Atlassian Rovo MCP Server...")
    async with MultiServerMCPClient(
      {
        "atlassian": {
          "transport": "stdio",
          "command": "npx.cmd",
          "args": [
            "-y",
            "mcp-atlassian"
          ],
          "env": {
            "ATLASSIAN_BASE_URL": os.environ.get("ATLASSIAN_URL"),
            "ATLASSIAN_EMAIL": os.environ.get("ATLASSIAN_USER_EMAIL"),
            "ATLASSIAN_API_TOKEN": os.environ.get("ATLASSIAN_API_TOKEN"),
            "PATH": os.environ.get("PATH")
          }
        }
      }
    ) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize connection handshake
            await session.initialize()
            
            print("Executing getAccessibleAtlassianResources...")
            # Execute tool with empty parameters as specified by protocol
            response = await session.call_tool("getAccessibleAtlassianResources", arguments={})
            
            # 4. Print out the raw dictionary response object
            print("\n--- Server Tool Output ---")
            print(response.content)

# Standard Colab snippet wrapper to handle asyncio loop compilation

async def create_agent(model_str: str, tools: list):
    """Helper function to create a tool-calling agent using LangGraph."""
    # Extracts 'gpt-4o-mini' from 'openai:gpt-4o-mini'
    model_name = model_str.split(":")[-1] if ":" in model_str else model_str
    llm = ChatOpenAI(model=model_name)
    jira_agent = llm.bind_tools(tools)
    return jira_agent

# Apply the patch once globally (perfect for Jupyter)
async def safe_run(client: MultiServerMCPClient):
  # Standard terminal execution
  return await client.get_tools()
    # else:
    #     # Jupyter environment: routes execution to a separate thread loop
    #     with concurrent.futures.ThreadPoolExecutor(1) as pool:
    #         return pool.submit(asyncio.run, coro).result()

async def initialize_mcp():
    # Target the proxy wrapper running on localhost inside Colab
    client = MultiServerMCPClient({
        "rovo-wrapped": {
            "transport": "http",          # Tell LangChain to access via HTTP/SSE bridge
            "url": "http://localhost:8080/mcp" # Route through the active mcp-remote instance
        }
    })
    
    # Retrieve the clean, structured tools
    tools = await client.get_tools()
    print(f"Successfully loaded {len(tools)} tools from Rovo!")
    return client, tools

# Run the async initialization block in Colab's event loop
        
async def search_jira_for_priority(feature_name: str) -> int:
  jirapriorities = 2
  # intialize client
  client = MultiServerMCPClient (
    {
      "atlassian": {
        "transport": "stdio",
        "command": "npx.cmd",
        "args": [
          "-y",
          "mcp-remote@latest",
          "https://mcp.atlassian.com/v1/mcp"
        ],  
        "env": {
          "ATLASSIAN_BASE_URL": os.environ.get("ATLASSIAN_URL"),
          "ATLASSIAN_EMAIL": os.environ.get("ATLASSIAN_USER_EMAIL"),
          "ATLASSIAN_API_TOKEN": os.environ.get("ATLASSIAN_API_TOKEN"),
          "PATH": os.environ.get("PATH")
        }
      }
    }
    )
  # client, tools = tragedies = await initialize_mcp()
  if (client):
      # Offload the blocking function to a background thread automatically
    tools = await safe_run(client)
    if(not len(tools)):
      print ("tools failure")
    # 3. Locate the "get_accessible_resources" tool (often named like atlassian-get_accessible_resources)
      # if (target_tool):
      #   print ({await target_tool.ainvoke({})})
      # else:
      #   print ("target tool not found")
      #tools = asyncio.run(client.get_tools(),debug='false')
      
    agent1 = await create_agent("openai:gpt-4o-mini", tools)
      # agent_input_message = HumanMessage(content=(
      #   "Search for string" f"{feature_name}""in space TeamProductManager and return the count of instances\n"
      # ))
      #llm_with_tools = llm.bind_tools(tools)
      #response = await agent1.ainvoke(f"Search for the word {feature_name} in space TeamProductManager and return a popularity score for the feature based on the number of issues found against it. The popularity score should be between 1 and 10 with 1 being lowest", tools)
      #response = await agent1.ainvoke(f"Search for the word {feature_name} in space TeamProductManager and return a popularity score for the feature based on the number of issues found against it. The popularity score should be between 1 and 10 with 1 being lowest")
    messages = [
      SystemMessage(content=f"You are an assistant with access to Jira. When interacting with the Atlassian MCP server, use the site id: {os.environ.get('ATLASSIAN_SITE_ID')}. Do not discover resources using getAccessibleAPIResources; directly execute searchJiraIssues using this target cloudid. Also do not use escape characters, that is the character '\' before quotesAlso, perform a wildcard search for each word input in all Projects and return the total count. For example for input 'User Authentication' input search for ~'User*' OR ~'Authentication*'. Set maxresults at 100"),
      HumanMessage(content=f"Search for the word like {feature_name} in issues in jira space TeamProductManager. Finally return a score between 1 and 10 based on the numbe of issues found. If the issues are less than 3 it will get a lower score between 1 and 3, rest of the scores you can decide, as the max count is 100. Create the output with the format Score= and Reasoning=")
    ] 
    response = await agent1.ainvoke(messages)
    data = response.model_dump_json(indent=2)    # Check if the model wants to call a tool
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # Find the matching tool in the tools list
            selected_tool = next(t for t in tools if t.name == tool_call["name"])
            # Execute the tool call
            if(selected_tool):
              tool_output = await selected_tool.ainvoke(tool_call["args"])
            # Add the tool output to the conversation
            messages.append(
                HumanMessage(content=f"Tool {tool_call['name']} returned: {tool_output}")
            )
        # Get the final response from the LLM after tool execution
        response = await agent1.ainvoke(messages)
        data = response.model_dump_json(indent=2)

    if (not data):
        print("No data returned")
    response_content = response.content
    #this is for debugging purposes. Edit this out if you want to output all the jira_priorities
    print(f"Value {response_content}")
    try:
            # Extract numeric value from LLM response content
            match = re.search(r"Score=(\d+)", str(response_content))
            jirapriorities = int(float(match.group(1))) if match else 3
            
            if jirapriorities > 10 or jirapriorities < 1:
                print (f"priority out of bounds") 
                jirapriorities = 3  #default
    except (ValueError, TypeError):
            print (f"count not found")
            jirapriorities = 3
      #return jirapriorities
  else:
      """Placeholder function to simulate searching JIRA for a feature's priority.
      In a real scenario, this would interact with the JIRA API.
      Returns a priority score between 1 and 10, or None if not found.
      """
      print(f"Setting default priority of feature: {feature_name}...")
      # Simulate JIRA API call with some hardcoded values for demonstration
      jira_priorities = {
        "Provide secure user login via SSO": 9, # High priority
        "Manage asset importance scoring": 6,  # Medium priority
        "Export Exposure Management Reports": 8, # High priority
        "Compute Breach Impact from exposure": 7, # High priority
        "Provide customization option for risk score": 9, # Critical priority
        "Provide vulnerability attack blocking": 5 # Medium priority
      }
      jirapriorities = jira_priorities.get(feature_name)
      print(f"Jira count found for: {feature_name}...{jirapriorities}")
  return jirapriorities

async def main():
    feature_list = {
        "Provide secure user login via SSO": 9, # High priority
        "Manage asset importance scoring": 6,  # Medium priority
        "Export Exposure Management Reports": 8, # High priority
        "Compute Breach Impact from exposure": 7, # High priority
        "Provide customization option for risk score": 9, # Critical priority
        "Provide vulnerability attack blocking": 5 # Medium priority
      }
    for f in feature_list:
      jira_value = await search_jira_for_priority(f)
      print (f"\"{f}\": {jira_value}")
    # try:
    #   await run_atlassian_tool()
    # except RuntimeError:
    # # Fallback structure if event loop is already occupied
    #   asyncio.get_event_loop().create_task(run_atlassian_tool())

if __name__ == "__main__":
   load_dotenv()
   asyncio.run(main())
