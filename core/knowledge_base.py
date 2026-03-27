"""
knowledge_base.py — Curated plant disease knowledge base with ChromaDB vector store.

Architecture:
    28 expert-crafted disease profiles (always available, O(1) lookup)
    +
    ChromaDB semantic index (for fuzzy/general queries)

Priority chain for retrieval:
    1. Exact dict lookup  (fastest, always hits for our 28 classes)
    2. ChromaDB semantic  (fuzzy match, general questions)
    3. Caller falls back to Wikipedia / Groq  (handled in tools.py)
"""

import os
from typing import Optional, List
import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------------
# Curated disease profiles — expert-written, factually grounded
# One entry per PLANT_CLASSES label.
# ---------------------------------------------------------------------------
DISEASE_PROFILES: dict[str, dict] = {

    # ── CORN ──────────────────────────────────────────────────────────────
    "Corn - Gray Leaf Spot": {
        "cause": "Caused by the fungus Cercospora zeae-maydis. Thrives in warm, humid conditions with poor air circulation and prolonged leaf wetness.",
        "symptoms": [
            "Rectangular, tan to gray lesions with distinct parallel edges running parallel to leaf veins",
            "Lesions are 1–6 cm long, initially water-soaked then turning necrotic gray-brown",
            "Severe infection causes premature leaf death and stalk rot",
        ],
        "treatment": [
            "Apply foliar fungicides (azoxystrobin, pyraclostrobin) at early tassel stage",
            "Remove and destroy infected plant debris after harvest",
            "Improve field drainage and increase row spacing to reduce humidity",
        ],
        "prevention": [
            "Plant resistant hybrids rated for gray leaf spot tolerance",
            "Rotate crops with non-host species (soybean, small grains) for at least one season",
        ],
        "pathogen": "Cercospora zeae-maydis",
        "severity": "High — can cause 50%+ yield loss in susceptible hybrids under favorable conditions",
    },

    "Corn - Common Rust": {
        "cause": "Caused by the obligate fungal pathogen Puccinia sorghi. Spreads via wind-dispersed urediniospores; favors cool temperatures (16–23°C) and high humidity.",
        "symptoms": [
            "Circular to elongated, brick-red to dark-brown pustules (uredinia) scattered on both leaf surfaces",
            "Pustules rupture through the epidermis, releasing powdery cinnamon-brown spores",
            "Severe infection causes yellowing, premature senescence, and reduced grain fill",
        ],
        "treatment": [
            "Apply triazole or strobilurin fungicides (propiconazole, azoxystrobin) at early pustule appearance",
            "Prioritize treatment on sweet corn and susceptible hybrids",
            "Monitor fields weekly during cool, wet periods during vegetative growth",
        ],
        "prevention": [
            "Select rust-resistant hybrids — check resistance ratings from your seed supplier",
            "Avoid late planting that exposes crops to peak spore periods",
        ],
        "pathogen": "Puccinia sorghi",
        "severity": "Moderate to High — primarily a concern in tropical and subtropical regions",
    },

    "Corn - Northern Leaf Blight": {
        "cause": "Caused by Exserohilum turcicum (Setosphaeria turcica). Spreads via wind and rain splash; favors moderate temperatures (18–27°C) and high humidity.",
        "symptoms": [
            "Long, elliptical, gray-green to tan lesions (2.5–15 cm) running parallel to leaf edges",
            "Lesions develop dark, sooty sporulation under humid conditions",
            "Lower leaves are infected first; disease progresses upward rapidly",
        ],
        "treatment": [
            "Apply foliar fungicides containing propiconazole or pyraclostrobin at early infection stages",
            "Destroy infected crop residue by deep tillage or burning post-harvest",
            "Ensure balanced nitrogen fertilization — excessive N promotes lush, susceptible tissue",
        ],
        "prevention": [
            "Use hybrids with Ht1, Ht2, Ht3 resistance genes for partial resistance",
            "Rotate with non-grass crops to reduce inoculum levels in soil",
        ],
        "pathogen": "Exserohilum turcicum",
        "severity": "High — can cause 30–60% yield loss when infection occurs before silking",
    },

    "Corn - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Leaves are uniformly green with no lesions, spots, or discoloration"],
        "treatment": ["No treatment required — maintain standard agronomic practices"],
        "prevention": [
            "Continue crop rotation and balanced fertilization",
            "Monitor regularly for early signs of disease or pest pressure",
        ],
        "pathogen": "None",
        "severity": "None — plant is healthy",
    },

    # ── GRAPE ─────────────────────────────────────────────────────────────
    "Grape - Black Rot": {
        "cause": "Caused by the ascomycete fungus Guignardia bidwellii. Overwinters in mummified fruit and infected wood; releases spores during rainfall in spring.",
        "symptoms": [
            "Small, circular, tan-brown lesions with dark borders on leaves",
            "Infected berries turn brown, shrivel, and become hard black mummies",
            "Dark pycnidia (fruiting bodies) visible as black dots in lesions",
        ],
        "treatment": [
            "Apply fungicides (mancozeb, myclobutanil, captan) from shoot emergence through veraison",
            "Remove and destroy all mummified fruit from the vine and ground immediately",
            "Prune to improve canopy airflow during dormancy",
        ],
        "prevention": [
            "Remove all mummified berries and diseased canes as primary inoculum control",
            "Apply protective fungicide program starting at 2.5 cm shoot growth",
        ],
        "pathogen": "Guignardia bidwellii",
        "severity": "Very High — can destroy 100% of fruit crop in susceptible varieties under wet conditions",
    },

    "Grape - Black Measles": {
        "cause": "A complex of wood-rotting fungi including Phaeoacremonium minimum and Phaeomoniella chlamydospora. Infects through pruning wounds; accumulates toxins over years.",
        "symptoms": [
            "Interveinal chlorosis (yellowing) with reddish-brown margins on leaves — classic 'tiger-stripe' pattern",
            "Berries develop small dark spots with white halos (bird's-eye rot pattern)",
            "Internal wood shows dark brown streaking when canes are cross-sectioned",
        ],
        "treatment": [
            "Remove and burn severely infected vines — no curative chemical treatment exists",
            "Paint pruning wounds immediately with fungicide (thiophanate-methyl paste) within 30 minutes of cutting",
            "Apply 2% sodium arsenite (where legal) as a preventive trunk treatment",
        ],
        "prevention": [
            "Prune during dry weather and immediately protect all wounds with fungicide paste",
            "Avoid large pruning wounds — double pruning (leaving spurs) reduces infection risk",
        ],
        "pathogen": "Phaeoacremonium spp., Phaeomoniella chlamydospora",
        "severity": "High — a chronic, incurable vine disease that reduces productivity over decades",
    },

    "Grape - Leaf Blight": {
        "cause": "Caused by Pseudocercospora vitis (isariopsis leaf spot) or Plasmopara viticola (downy mildew). Favors warm, humid conditions and dense canopies.",
        "symptoms": [
            "Irregular, dark brown necrotic patches on older leaves, often starting at margins",
            "Yellow halo surrounding necrotic areas visible in early infection",
            "Severe cases cause premature defoliation weakening vine and reducing fruit quality",
        ],
        "treatment": [
            "Apply copper-based fungicides (copper hydroxide) or mancozeb at disease onset",
            "Remove severely infected leaves to reduce spread within the canopy",
            "Reduce overhead irrigation and improve canopy management for airflow",
        ],
        "prevention": [
            "Maintain open canopy through shoot positioning and leaf removal",
            "Apply preventive fungicide program during periods of prolonged leaf wetness",
        ],
        "pathogen": "Pseudocercospora vitis / Plasmopara viticola",
        "severity": "Moderate — primarily impacts vine vigor and subsequent season performance",
    },

    "Grape - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Leaves are uniformly green with no spots, discoloration, or blight"],
        "treatment": ["No treatment required — continue standard vineyard management"],
        "prevention": [
            "Maintain balanced pruning for open canopy structure",
            "Apply preventive fungicide program at bud break as a precaution",
        ],
        "pathogen": "None",
        "severity": "None — vine is healthy",
    },

    # ── PEACH ─────────────────────────────────────────────────────────────
    "Peach - Bacterial Spot": {
        "cause": "Caused by Xanthomonas arboricola pv. pruni. Spreads via rain splash and wind during wet spring weather; enters through stomata and wounds.",
        "symptoms": [
            "Small, water-soaked angular spots on leaves that turn purple-brown with yellow halos",
            "Infected tissue drops out creating a 'shot-hole' appearance",
            "Fruit develops shallow, sunken, water-soaked lesions that crack and ooze gum",
        ],
        "treatment": [
            "Apply copper bactericides (copper hydroxide, copper octanoate) from bud swell through petal fall",
            "Prune infected shoots beyond the visible lesion margin during dry weather",
            "Avoid overhead irrigation — use drip irrigation to reduce leaf wetness",
        ],
        "prevention": [
            "Select resistant peach varieties (e.g., Contender, Reliance) for your region",
            "Apply dormant copper spray in late winter before bud swell",
        ],
        "pathogen": "Xanthomonas arboricola pv. pruni",
        "severity": "High — major commercial disease causing fruit and leaf damage across stone fruit industries",
    },

    "Peach - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Leaves are uniformly green without spots, shot-holes, or discoloration"],
        "treatment": ["No treatment required — maintain standard orchard practices"],
        "prevention": [
            "Apply preventive copper sprays during dormancy as a precaution",
            "Monitor for early bacterial spot symptoms during wet spring conditions",
        ],
        "pathogen": "None",
        "severity": "None — tree is healthy",
    },

    # ── PEPPER BELL ───────────────────────────────────────────────────────
    "Pepper Bell - Bacterial Spot": {
        "cause": "Caused by Xanthomonas euvesicatoria (and related species). Spreads via infected seed, rain splash, and contact; thrives in warm, wet conditions above 24°C.",
        "symptoms": [
            "Small, water-soaked spots on leaves that enlarge into irregular brown lesions with yellow halos",
            "Lesion centers may drop out ('shot-hole') in dry conditions",
            "Fruit develops raised, scabby lesions that make it unmarketable",
        ],
        "treatment": [
            "Spray copper hydroxide + mancozeb (tank mix) every 5–7 days during wet conditions",
            "Remove and destroy heavily infected plant material immediately",
            "Stop overhead irrigation — shift to drip/furrow systems",
        ],
        "prevention": [
            "Use certified disease-free seed and resistant varieties (e.g., Aristotle, Revolution)",
            "Do not work in fields when plants are wet — bacteria spread on hands and tools",
        ],
        "pathogen": "Xanthomonas euvesicatoria",
        "severity": "High — can cause total crop loss under warm, wet conditions with susceptible varieties",
    },

    "Pepper Bell - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Plant is vigorous with uniformly green, spot-free foliage"],
        "treatment": ["No treatment required"],
        "prevention": [
            "Use disease-free certified seed for every planting",
            "Practice 2-3 year crop rotation away from solanaceous crops",
        ],
        "pathogen": "None",
        "severity": "None — plant is healthy",
    },

    # ── POTATO ────────────────────────────────────────────────────────────
    "Potato - Early Blight": {
        "cause": "Caused by Alternaria solani. A necrotrophic fungus that primarily attacks older, stressed tissues; favors warm temperatures (24–29°C) with alternating wet/dry periods.",
        "symptoms": [
            "Dark-brown, circular lesions with distinctive concentric rings (target-board pattern)",
            "Lesions first appear on lower, older leaves and progress upward",
            "Yellow halo surrounds each lesion; severely infected leaves yellow and drop prematurely",
        ],
        "treatment": [
            "Apply fungicides containing chlorothalonil, azoxystrobin, or mancozeb at first symptom appearance",
            "Ensure adequate potassium fertilization — deficiency increases susceptibility",
            "Remove and destroy infected lower leaves to slow upward spread",
        ],
        "prevention": [
            "Use certified disease-free seed tubers from reputable suppliers",
            "Maintain adequate soil nutrition — early blight is an opportunistic pathogen of stressed plants",
        ],
        "pathogen": "Alternaria solani",
        "severity": "Moderate — primarily quality and defoliation issue; rarely kills plants outright",
    },

    "Potato - Late Blight": {
        "cause": "Caused by Phytophthora infestans, an oomycete (water mold). Highly destructive; caused the Irish Famine (1845). Spreads extremely rapidly in cool, moist conditions (10–20°C).",
        "symptoms": [
            "Pale green to brown, water-soaked lesions on leaf edges and tips that expand rapidly",
            "White, cottony sporulation visible on lesion undersides under humid conditions",
            "Infected tubers show brown, granular rot that spreads to whole storage",
        ],
        "treatment": [
            "Apply systemic fungicides (metalaxyl, cymoxanil) + contact protectant (mancozeb) IMMEDIATELY at first sign",
            "Destroy and bury all infected plant material — do not compost",
            "Harvest tubers in dry conditions; cure properly before storage",
        ],
        "prevention": [
            "Plant certified blight-free seed tubers and resistant varieties (e.g., Defender, Sarpo Mira)",
            "Apply preventive fungicide program during cool, wet weather before symptoms appear",
        ],
        "pathogen": "Phytophthora infestans",
        "severity": "CRITICAL — can destroy an entire field in 7–10 days under favorable conditions. Act immediately.",
    },

    "Potato - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Plants are vigorously growing with uniformly green, spot-free foliage"],
        "treatment": ["No treatment required"],
        "prevention": [
            "Maintain preventive fungicide program during cool, wet weather (late blight risk periods)",
            "Use certified seed tubers every season",
        ],
        "pathogen": "None",
        "severity": "None — plant is healthy",
    },

    # ── RICE ──────────────────────────────────────────────────────────────
    "Rice - Brown Spot": {
        "cause": "Caused by Cochliobolus miyabeanus (anamorph: Bipolaris oryzae). Associated with nutrient-deficient soils, especially low silicon and potassium; favors warm, humid conditions.",
        "symptoms": [
            "Oval to circular brown spots (1–5 mm) with a dark-brown border and gray center on leaves",
            "Spots may coalesce causing large necrotic areas on severely infected leaves",
            "Infected grain shows discoloration and reduced seed quality ('pecky rice')",
        ],
        "treatment": [
            "Apply foliar fungicides (iprodione, carbendazim, or mancozeb) at early tillering and boot stage",
            "Apply potassium and silicon fertilizers to correct nutritional deficiencies",
            "Treat seed with systemic fungicide (carboxin) before planting",
        ],
        "prevention": [
            "Correct soil nutrient deficiencies — balanced NPK + silicon application reduces severity dramatically",
            "Use resistant varieties and disease-free certified seed",
        ],
        "pathogen": "Cochliobolus miyabeanus",
        "severity": "High — caused the Bengal Famine of 1943. Nutrient correction is the most effective management tool.",
    },

    "Rice - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Crop is uniformly green and growing vigorously with no spots or discoloration"],
        "treatment": ["No treatment required — maintain balanced fertilization"],
        "prevention": [
            "Monitor weekly for early brown spot and leaf blast symptoms",
            "Maintain adequate silicon and potassium nutrition as preventive measure",
        ],
        "pathogen": "None",
        "severity": "None — crop is healthy",
    },

    "Rice - Hispa": {
        "cause": "Caused by an insect pest — the Rice Hispa beetle (Dicladispa armigera). Adult beetles scrape the upper leaf surface; larvae mine inside leaves.",
        "symptoms": [
            "White, irregular, parallel streaks on leaves caused by adult scraping feeding",
            "Brown blotches and mines inside leaves visible when held against light (larval damage)",
            "Severe infestation turns leaves white and papery, leading to significant yield loss",
        ],
        "treatment": [
            "Clip and destroy leaf tips containing larvae to prevent adult emergence",
            "Apply insecticides (chlorpyrifos, imidacloprid) at 1–2 beetles per hill threshold",
            "Release Tetrastichus hisparum (parasitoid wasp) for biological control",
        ],
        "prevention": [
            "Avoid dense planting — hispa thrives in shaded, humid canopies",
            "Flood paddy water during early infestation to drown larvae",
        ],
        "pathogen": "Dicladispa armigera (insect pest — not a fungal/bacterial disease)",
        "severity": "Moderate to High — particularly damaging during seedling and tillering stages",
    },

    "Rice - Leaf Blast": {
        "cause": "Caused by Magnaporthe oryzae (blast fungus). One of the most destructive rice diseases globally. Favors high nitrogen, silicon deficiency, and cool nights with warm days.",
        "symptoms": [
            "Diamond-shaped lesions with gray-white centers and brown-red borders on leaves",
            "Lesions at the node (node blast) show blackening and breaking of the stem",
            "Neck blast causes whitening of the panicle ('dead neck') with complete grain sterility",
        ],
        "treatment": [
            "Apply tricyclazole, isoprothiolane, or azoxystrobin fungicide immediately at first lesion appearance",
            "Reduce nitrogen application — excessive N dramatically increases blast severity",
            "Drain fields for 3–5 days to reduce humidity at the canopy",
        ],
        "prevention": [
            "Plant blast-resistant varieties (check national seed board ratings for your region)",
            "Apply silicon fertilizer (silica slag) — silicon strengthens cell walls against fungal penetration",
        ],
        "pathogen": "Magnaporthe oryzae",
        "severity": "CRITICAL — can destroy 100% of grain yield if neck blast occurs at heading stage.",
    },

    # ── STRAWBERRY ────────────────────────────────────────────────────────
    "Strawberry - Leaf Scorch": {
        "cause": "Caused by Diplocarpon earlianum. Spreads via rain splash; favors warm, wet conditions. Distinct from drought scorch — this is a true fungal disease.",
        "symptoms": [
            "Small, irregular purple-red spots on upper leaf surface, scattered across the leaf blade",
            "Spots lack yellow halos (distinguishing from angular leaf spot); centers remain dark purple",
            "Severely infected leaves turn bronze-red ('scorched') and die, reducing photosynthesis",
        ],
        "treatment": [
            "Apply fungicides (captan, myclobutanil, or azoxystrobin) every 7–10 days during wet conditions",
            "Remove and destroy old infected leaves after harvest to reduce inoculum",
            "Avoid overhead irrigation — use drip irrigation to keep foliage dry",
        ],
        "prevention": [
            "Plant resistant varieties (e.g., Chandler, Camarosa show moderate resistance)",
            "Renovate planting after harvest by mowing and thinning to improve airflow",
        ],
        "pathogen": "Diplocarpon earlianum",
        "severity": "Moderate — primarily a foliage disease but chronic infection weakens plants and reduces runner production",
    },

    "Strawberry - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Plants are healthy with uniformly green, unblemished foliage and normal growth"],
        "treatment": ["No treatment required"],
        "prevention": [
            "Apply preventive fungicide program during prolonged wet periods",
            "Maintain proper plant spacing for adequate air circulation",
        ],
        "pathogen": "None",
        "severity": "None — plant is healthy",
    },

    # ── TOMATO ────────────────────────────────────────────────────────────
    "Tomato - Bacterial Spot": {
        "cause": "Caused by Xanthomonas euvesicatoria and related species. Spreads via contaminated seed, rain splash, and infected transplants; highly favored by warm, wet conditions.",
        "symptoms": [
            "Small, water-soaked circular spots on leaves that turn brown with yellow halos",
            "Spots may coalesce, causing large necrotic areas and premature leaf drop",
            "Fruit develops raised, scabby, tan spots that render it unmarketable",
        ],
        "treatment": [
            "Apply copper bactericide (copper hydroxide) + mancozeb tank mix every 5–7 days during wet weather",
            "Remove and destroy heavily infected plant material immediately",
            "Avoid working in fields when foliage is wet",
        ],
        "prevention": [
            "Use certified disease-free seed and resistant varieties (e.g., Plum Regal, Shanty)",
            "Practice 2-year crop rotation away from all solanaceous crops",
        ],
        "pathogen": "Xanthomonas euvesicatoria",
        "severity": "High — widespread in tropical and subtropical tomato production regions",
    },

    "Tomato - Early Blight": {
        "cause": "Caused by Alternaria solani. Opportunistic necrotrophic fungus; infects older/stressed tissue first. Favors warm, alternating wet/dry cycles.",
        "symptoms": [
            "Dark brown concentric ring (target-board) lesions on older lower leaves",
            "Yellow chlorotic halo surrounds each lesion",
            "Stem lesions ('collar rot') at the soil line cause seedling damping-off",
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, azoxystrobin, or mancozeb) every 7–10 days from early infection",
            "Mulch soil to prevent rain-splash spread of soilborne inoculum",
            "Stake plants to improve airflow and reduce contact with soil",
        ],
        "prevention": [
            "Use resistant varieties and practice 2-3 year rotation with non-solanaceous crops",
            "Ensure balanced potassium nutrition — deficiency severely increases susceptibility",
        ],
        "pathogen": "Alternaria solani",
        "severity": "Moderate to High — progressive defoliation reduces fruit size and yield significantly",
    },

    "Tomato - Late Blight": {
        "cause": "Caused by Phytophthora infestans. Extremely aggressive oomycete. Spreads rapidly in cool (10–20°C), wet conditions. Same pathogen as Potato Late Blight.",
        "symptoms": [
            "Dark green to brown, greasy-looking water-soaked lesions on leaves and stems",
            "White sporulation on lower leaf surface under humid conditions",
            "Fruit shows greenish-brown, firm rot that spreads through the entire fruit rapidly",
        ],
        "treatment": [
            "Apply systemic fungicides (metalaxyl, dimethomorph) immediately — do not wait",
            "Tank mix with contact fungicide (mancozeb, chlorothalonil) for complete coverage",
            "Remove and destroy all infected material — bag and dispose, do not compost",
        ],
        "prevention": [
            "Plant resistant varieties (FHIA, Mountain Merit, Plum Regal)",
            "Apply preventive fungicide program during cool, wet weather before symptoms appear",
        ],
        "pathogen": "Phytophthora infestans",
        "severity": "CRITICAL — can destroy a field within 1–2 weeks. Immediate action is required.",
    },

    "Tomato - Leaf Mold": {
        "cause": "Caused by Passalora fulva (formerly Fulvia fulva). A greenhouse specialist disease; thrives in high humidity (>85%) and temperatures of 22–25°C.",
        "symptoms": [
            "Pale green to yellow spots on the upper leaf surface without clear margins",
            "Olive-green to gray-brown velvety sporulation on the lower leaf surface under spots",
            "Severely infected leaves curl upward, become chlorotic, and drop prematurely",
        ],
        "treatment": [
            "Reduce greenhouse humidity below 85% immediately — increase ventilation and heating",
            "Apply fungicides (chlorothalonil, mancozeb, or azoxystrobin) at first symptom appearance",
            "Remove and destroy infected leaves to stop sporulation",
        ],
        "prevention": [
            "Maintain relative humidity below 85% through active ventilation and spacing",
            "Plant resistant varieties with Cf resistance genes (Cf-4, Cf-9) for greenhouse production",
        ],
        "pathogen": "Passalora fulva",
        "severity": "High in greenhouses — rarely severe in field conditions with adequate airflow",
    },

    "Tomato - Septoria Leaf Spot": {
        "cause": "Caused by Septoria lycopersici. Soilborne fungus with very long survival in infected debris; spreads by rain splash. Favors warm (20–25°C), wet conditions.",
        "symptoms": [
            "Circular spots (3–5 mm) with dark brown borders, tan-gray centers on lower leaves",
            "Tiny black pycnidia (spore-producing structures) visible in lesion centers with a hand lens",
            "Disease progresses rapidly upward; plants may be completely defoliated in 2–3 weeks",
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb, or azoxystrobin) every 7–10 days",
            "Mulch soil surface to reduce rain-splash transmission from infected debris",
            "Remove infected lower leaves and dispose of immediately",
        ],
        "prevention": [
            "Practice 2-3 year rotation away from tomato, potato, and other solanaceous crops",
            "Stake plants and train for airflow — Septoria thrives in dense, humid canopies",
        ],
        "pathogen": "Septoria lycopersici",
        "severity": "High — one of the most common and damaging tomato foliage diseases worldwide",
    },

    "Tomato - Spider Mites": {
        "cause": "Caused by Tetranychus urticae (two-spotted spider mite). An arthropod pest — not a fungal or bacterial disease. Thrives in hot, dry, dusty conditions.",
        "symptoms": [
            "Fine yellow stippling (pin-prick dots) on upper leaf surface caused by mite feeding",
            "Fine, silky webbing on undersides of leaves and between leaflets in heavy infestations",
            "Leaves turn bronzed, yellow, then brown and die — plant appears drought-stressed",
        ],
        "treatment": [
            "Apply miticide (abamectin, bifenazate, spiromesifen) — rotate modes of action to prevent resistance",
            "Release predatory mites (Phytoseiulus persimilis) for biological control in greenhouses",
            "Strong water jet spray on leaf undersides disrupts colonies effectively",
        ],
        "prevention": [
            "Maintain adequate irrigation — water-stressed plants are significantly more susceptible",
            "Avoid excessive nitrogen and broad-spectrum insecticides that kill natural predators",
        ],
        "pathogen": "Tetranychus urticae (arthropod pest — not a disease)",
        "severity": "High in hot, dry conditions — populations can explode in 5–7 days under drought stress",
    },

    "Tomato - Healthy": {
        "cause": "No disease detected.",
        "symptoms": ["Plant is vigorous with dark green, unblemished foliage and normal growth"],
        "treatment": ["No treatment required"],
        "prevention": [
            "Maintain regular scouting for early disease and pest detection",
            "Apply preventive fungicide program during warm, wet weather",
        ],
        "pathogen": "None",
        "severity": "None — plant is healthy",
    },
}


# ---------------------------------------------------------------------------
# ChromaDB semantic index — built once, persisted to disk
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", ".chromadb_knowledge")
_COLLECTION_NAME = "plant_disease_kb"
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    """Lazy-load or create the ChromaDB collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=_DB_PATH)
    existing = [c.name for c in _client.list_collections()]

    if _COLLECTION_NAME in existing:
        _collection = _client.get_collection(_COLLECTION_NAME)
        return _collection

    # First run — build the index
    print("[KnowledgeBase] Building ChromaDB index for the first time...")
    _collection = _client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hf:model": "all-MiniLM-L6-v2"},
    )

    documents, ids, metadatas = [], [], []
    for disease, profile in DISEASE_PROFILES.items():
        # Build a rich text document for each disease
        doc = (
            f"Disease: {disease}\n"
            f"Pathogen: {profile['pathogen']}\n"
            f"Cause: {profile['cause']}\n"
            f"Symptoms: {'; '.join(profile['symptoms'])}\n"
            f"Treatment: {'; '.join(profile['treatment'])}\n"
            f"Prevention: {'; '.join(profile['prevention'])}\n"
            f"Severity: {profile['severity']}"
        )
        documents.append(doc)
        ids.append(disease.replace(" ", "_").replace("-", "_"))
        metadatas.append({"disease": disease})

    _collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"[KnowledgeBase] Indexed {len(documents)} disease profiles.")
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_disease(disease_name: str) -> Optional[dict]:
    """
    Fast O(1) exact lookup for known disease names.
    Returns the full profile dict, or None if not found.
    """
    return DISEASE_PROFILES.get(disease_name)


def semantic_search(query: str, n_results: int = 3) -> List[str]:
    """
    Semantic similarity search over the knowledge base.
    Used for general questions that aren't exact disease name lookups.
    Returns list of relevant document strings.
    """
    try:
        collection = _get_collection()
        results = collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[KnowledgeBase] ChromaDB semantic search failed: {e}")
        return []


def format_profile(profile: dict) -> str:
    """Convert a disease profile dict into a structured text string."""
    symptoms_text = "\n".join(f"  • {s}" for s in profile["symptoms"])
    treatment_text = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(profile["treatment"]))
    prevention_text = "\n".join(f"  • {p}" for p in profile["prevention"])

    return (
        f"PATHOGEN: {profile['pathogen']}\n\n"
        f"CAUSE: {profile['cause']}\n\n"
        f"SYMPTOMS:\n{symptoms_text}\n\n"
        f"TREATMENT:\n{treatment_text}\n\n"
        f"PREVENTION:\n{prevention_text}\n\n"
        f"SEVERITY: {profile['severity']}"
    )
