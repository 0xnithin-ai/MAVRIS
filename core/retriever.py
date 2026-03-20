from langchain.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

class DiseaseRetrieverTool:
    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
        
    def fetch_info(self, disease_name: str, confidence_score: float) -> str:
        if confidence_score < 0.3:
            return "Confidence is too low to perform accurate retrieval. Need an improved image."
            
        print(f"Fetching trusted info on {disease_name}")
        query = f"plant disease {disease_name} symptoms causes treatment summary"
        try:
            return self.search_tool.run(query)
        except Exception as e:
            return f"Failed to retrieve documentation: {e}"

