import os
import sys
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# 1. Create a dummy image
img_path = os.path.join(os.getcwd(), "dummy_leaf.jpg")
Image.new('RGB', (224, 224), color=(34, 139, 34)).save(img_path)  # Forest green image

# 2. Import the agent
print("[Test] Initializing PlantDiseaseAgent...")
from core.agent import PlantDiseaseAgent
agent = PlantDiseaseAgent()

# 3. Run the pipeline
print(f"[Test] Running Agent on {img_path}")
print("-" * 50)
try:
    response = agent.run("What disease does this leaf have?", image_path=img_path)
    print("\nFINAL AGENT RESPONSE:\n")
    print(response)
except Exception as e:
    print(f"\n[ERROR] Pipeline failed: {e}")

print("-" * 50)
print("[Test] Completed.")
