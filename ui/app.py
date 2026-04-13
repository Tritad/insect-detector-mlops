import base64
import os
from concurrent.futures import ThreadPoolExecutor
from html import escape

import requests
import streamlit as st

st.set_page_config(page_title="Pest InSight Classifier", layout="wide", page_icon="🐞")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Noto+Sans+Thai:wght@400;600;700&display=swap');

    :root {
        --green-900: #1b5e20;
        --green-700: #2e7d32;
        --green-500: #66a63f;
        --green-100: #eef8e7;
        --surface: rgba(255, 255, 255, 0.78);
        --surface-solid: #ffffff;
        --border-soft: rgba(27, 94, 32, 0.14);
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(255,255,255,0.60), transparent 45%),
            linear-gradient(145deg, #f7fbf2 0%, #e8f4dc 55%, #deedcf 100%);
    }

    .block-container {
        padding-top: 1.8rem;
        max-width: 1220px;
        padding-bottom: 1.25rem;
    }

    .hero-shell {
        max-width: 980px;
        margin: 0 auto 24px auto;
        border: 1px solid var(--border-soft);
        border-radius: 22px;
        padding: 28px 28px 24px 28px;
        background: var(--surface);
        backdrop-filter: blur(8px);
        box-shadow: 0 12px 28px rgba(16, 74, 26, 0.08);
        display: flex;
        flex-direction: column;
        align-items: center;
        animation: fadeSlideIn 420ms ease-out;
    }

    .hero-badge {
        width: fit-content;
        margin: 0 auto 12px auto;
        border-radius: 999px;
        border: 1px solid rgba(46, 125, 50, 0.22);
        background: #f5ffef;
        color: var(--green-700);
        padding: 7px 14px;
        font-family: 'Manrope', 'Noto Sans Thai', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    .hero-title {
        color: var(--green-900);
        font-family: 'Manrope', 'Noto Sans Thai', sans-serif;
        font-weight: 800;
        font-size: clamp(2rem, 4.2vw, 3.6rem);
        text-align: center;
        letter-spacing: -0.02em;
        line-height: 1.06;
        margin: 0;
    }

    .hero-subtitle {
        max-width: 760px;
        margin: 12px auto 0 auto;
        color: #39663a;
        width: 100%;
        text-align: center;
        font-family: 'Noto Sans Thai', sans-serif;
        font-size: clamp(1rem, 2vw, 1.2rem);
        line-height: 1.6;
        font-weight: 500;
    }

    .uploader-caption {
        margin: 8px 0 10px 2px;
        color: #2e5c2d;
        font-family: 'Noto Sans Thai', sans-serif;
        font-size: 0.98rem;
        font-weight: 600;
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(46, 125, 50, 0.35) !important;
        border-radius: 14px !important;
        background: rgba(255, 255, 255, 0.86) !important;
        transition: all 0.25s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(46, 125, 50, 0.7) !important;
        background: rgba(248, 255, 244, 0.95) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #2b5d2f;
        font-family: 'Noto Sans Thai', sans-serif;
    }

    @media (max-width: 768px) {
        .hero-shell {
            padding: 20px 16px 18px 16px;
            border-radius: 18px;
        }

        .hero-subtitle {
            line-height: 1.5;
        }
    }

    .result-card {
        background: var(--surface-solid);
        border-radius: 20px;
        border: 1px solid rgba(28, 84, 33, 0.10);
        box-shadow: 0 10px 24px rgba(19, 76, 26, 0.10);
        margin-bottom: 25px;
        overflow: hidden;
        animation: fadeSlideIn 360ms ease-out;
        transition: all 0.35s ease;
    }

    .result-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 18px 34px rgba(46, 125, 50, 0.22);
    }

    .result-image {
        width: 100%;
        display: block;
        height: 220px;
        object-fit: cover;
        border-radius: 20px 20px 0 0;
    }

    .card-info {
        padding: 20px;
        font-family: 'Noto Sans Thai', sans-serif;
        background: var(--surface-solid);
        border-radius: 0 0 20px 20px;
    }

    .species-label {
        color: var(--green-700);
        font-size: 1.12rem;
        font-weight: 700;
        display: block;
    }

    .conf-text { color: #666; font-size: 0.9rem; margin-top: 6px; }

    .conf-track {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        background: var(--green-100);
        overflow: hidden;
        margin-top: 10px;
    }

    .conf-fill {
        height: 100%;
        background: linear-gradient(45deg, var(--green-700), var(--green-500));
    }

    .advice-title {
        display: block;
        margin-top: 12px;
        color: #1f6d27;
        font-weight: 700;
    }

    .symptom-title {
        display: block;
        margin-top: 12px;
        color: #255a8a;
        font-weight: 700;
    }

    .symptom-text {
        margin-top: 6px;
        color: #334155;
        font-size: 0.84rem;
        line-height: 1.45;
        background: #f1f7ff;
        border: 1px solid rgba(37, 90, 138, 0.16);
        border-radius: 10px;
        padding: 8px 10px;
    }

    .advice-text {
        margin-top: 6px;
        color: #3f4b3f;
        font-size: 0.84rem;
        line-height: 1.45;
        background: #f6fff1;
        border: 1px solid rgba(46, 125, 50, 0.16);
        border-radius: 10px;
        padding: 8px 10px;
    }

    .advice-note {
        margin-top: 7px;
        color: #6a6f67;
        font-size: 0.76rem;
        line-height: 1.35;
    }

    .error-text {
        color: #c62828;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin: 10px 0 16px 0;
        padding: 12px 16px;
        border-radius: 14px;
        border: 1px solid rgba(46, 125, 50, 0.2);
        background: rgba(255, 255, 255, 0.80);
        backdrop-filter: blur(4px);
    }

    .result-title {
        margin: 0;
        color: var(--green-900);
        font-family: 'Manrope', 'Noto Sans Thai', sans-serif;
        font-size: clamp(1.25rem, 2.2vw, 1.85rem);
        font-weight: 800;
        line-height: 1.25;
    }

    .result-count {
        color: var(--green-700);
        border: 1px solid rgba(46, 125, 50, 0.18);
        background: #f6fff0;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.86rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .footer-note {
        margin-top: 22px;
        padding: 14px 12px 8px 12px;
        border-top: 1px solid rgba(27, 94, 32, 0.14);
        text-align: center;
        color: #5a6f59;
        font-size: 0.9rem;
        font-family: 'Manrope', 'Noto Sans Thai', sans-serif;
    }

    .empty-state {
        margin-top: 12px;
        border: 1px solid rgba(46, 125, 50, 0.18);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: 0 8px 22px rgba(22, 92, 36, 0.10);
        padding: 16px 18px;
        color: #2d5f31;
        font-family: 'Noto Sans Thai', sans-serif;
        font-size: 0.98rem;
        line-height: 1.5;
    }

    .empty-state strong {
        color: #1f6e2a;
        font-family: 'Manrope', 'Noto Sans Thai', sans-serif;
        font-weight: 800;
    }

    @media (max-width: 768px) {
        .result-header {
            flex-direction: column;
            align-items: flex-start;
        }
    }

    @keyframes fadeSlideIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border: 0;
        border-radius: 14px;
        background: linear-gradient(135deg, #2e7d32 0%, #5aaf47 100%);
        color: #ffffff;
        font-family: 'Manrope', 'Noto Sans Thai', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        box-shadow: 0 10px 22px rgba(46, 125, 50, 0.30);
        transition: transform 0.16s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(46, 125, 50, 0.34);
        filter: saturate(1.05);
    }

    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 8px 16px rgba(46, 125, 50, 0.26);
    }

    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(132, 193, 116, 0.45), 0 12px 24px rgba(46, 125, 50, 0.3);
    }

    .stButton > button:disabled {
        filter: grayscale(0.2);
        opacity: 0.72;
        cursor: not-allowed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

INSECT_LABELS_EN = [
    "rice leaf roller", "rice leaf caterpillar", "paddy stem maggot", "asiatic rice borer",
    "yellow rice borer", "rice gall midge", "Rice Stemfly", "brown plant hopper",
    "white backed plant hopper", "small brown plant hopper", "rice water weevil", "rice leafhopper",
    "grain spreader thrips", "rice shell pest", "grub", "mole cricket",
    "wireworm", "white margined moth", "black cutworm", "large cutworm",
    "yellow cutworm", "red spider", "corn borer", "army worm",
    "aphids", "Potosiabre vitarsis", "peach borer", "english grain aphid",
    "green bug", "bird cherry-oataphid", "wheat blossom midge", "penthaleus major",
    "longlegged spider mite", "wheat phloeothrips", "wheat sawfly", "cerodonta denticornis",
    "beet fly", "flea beetle", "cabbage army worm", "beet army worm",
    "Beet spot flies", "meadow moth", "beet weevil", "sericaorient alismots chulsky",
    "alfalfa weevil", "flax budworm", "alfalfa plant bug", "tarnished plant bug",
    "Locustoidea", "lytta polita", "legume blister beetle", "blister beetle",
    "therioaphis maculata Buckton", "odontothrips loti", "Thrips", "alfalfa seed chalcid",
    "Pieris canidia", "Apolygus lucorum", "Limacodidae", "Viteus vitifoliae",
    "Colomerus vitis", "Brevipoalpus lewisi McGregor", "oides decempunctata", "Polyphagotars onemus latus",
    "Pseudococcus comstocki Kuwana", "parathrene regalis", "Ampelophaga", "Lycorma delicatula",
    "Xylotrechus", "Cicadella viridis", "Miridae", "Trialeurodes vaporariorum",
    "Erythroneura apicalis", "Papilio xuthus", "Panonchus citri McGregor", "Phyllocoptes oleiverus ashmead",
    "Icerya purchasi Maskell", "Unaspis yanonensis", "Ceroplastes rubens", "Chrysomphalus aonidum",
    "Parlatoria zizyphus Lucus", "Nipaecoccus vastalor", "Aleurocanthus spiniferus", "Tetradacus c Bactrocera minax",
    "Dacus dorsalis(Hendel)", "Bactrocera tsuneonis", "Prodenia litura", "Adristyrannus",
    "Phyllocnistis citrella Stainton", "Toxoptera citricidus", "Toxoptera aurantii", "Aphis citricola Vander Goot",
    "Scirtothrips dorsalis Hood", "Dasineura sp", "Lawana imitata Melichar", "Salurnis marginella Guerr",
    "Deporaus marginatus Pascoe", "Chlumetia transversa", "Mango flat beak leafhopper", "Rhytidodera bowrinii white",
    "Sternochetus frigidus", "Cicadellidae",
]

INSECT_LABELS_TH = [
    "หนอนม้วนใบข้าว", "หนอนกินใบข้าว", "หนอนแมลงวันเจาะลำต้นข้าว", "หนอนเจาะลำต้นข้าวเอเชีย",
    "หนอนเจาะลำต้นข้าวสีเหลือง", "แมลงบั่วข้าว", "แมลงวันลำต้นข้าว", "เพลี้ยกระโดดสีน้ำตาล",
    "เพลี้ยกระโดดหลังขาว", "เพลี้ยกระโดดสีน้ำตาลเล็ก", "ด้วงงวงน้ำข้าว", "เพลี้ยจักจั่นข้าว",
    "เพลี้ยไฟกระจายเมล็ด", "แมลงศัตรูเปลือกข้าว", "หนอนด้วง", "จิ้งหรีดตุ่น",
    "หนอนลวด", "ผีเสื้อกลางคืนขอบขาว", "หนอนกระทู้ดำ", "หนอนกระทู้ใหญ่",
    "หนอนกระทู้เหลือง", "ไรแดง", "หนอนเจาะลำต้นข้าวโพด", "หนอนทหาร",
    "เพลี้ยอ่อน", "โปโตเซียเบร วิตาร์ซิส", "หนอนเจาะพีช", "เพลี้ยอ่อนธัญพืชอังกฤษ",
    "เพลี้ยเขียว", "เพลี้ยอ่อนเบิร์ดเชอร์รี-โอ๊ต", "แมลงบั่วดอกข้าวสาลี", "ไรเพนทาลิอุส เมเจอร์",
    "ไรแมงมุมขายาว", "เพลี้ยไฟข้าวสาลี", "แตนเลื่อยข้าวสาลี", "เซโรดอนตา เดนติคอร์นิส",
    "แมลงวันบีต", "ด้วงหมัด", "หนอนทหารกะหล่ำปลี", "หนอนทหารบีต",
    "แมลงวันจุดบีต", "ผีเสื้อทุ่งหญ้า", "ด้วงงวงบีต", "เซริกาโอเรียนต์ อลิสมอตส์ ชูลสกี",
    "ด้วงงวงอัลฟัลฟา", "หนอนเจาะตาลินิน", "แมลงมวนพืชอัลฟัลฟา", "แมลงมวนพืชลายด่าง",
    "ตั๊กแตนกลุ่ม Locustoidea", "ลิตตา โพลิตา", "ด้วงกดพืชตระกูลถั่ว", "ด้วงกด",
    "เทอริโออาฟิส มาคูลาตา", "โอดอนโตทริปส์ โลติ", "เพลี้ยไฟ", "แตนเบียนเมล็ดอัลฟัลฟา",
    "ผีเสื้อหนอนกะหล่ำ Pieris canidia", "อะโพลิกัส ลูโครัม", "หนอนคันไฟวงศ์ Limacodidae", "ไรองุ่น Viteus vitifoliae",
    "ไรองุ่นโคโลเมอรัส", "ไรรอบตาเลวิซี", "ออยเดส เดเซมพังค์ตาตา", "โพลีฟาโกทาร์โซเนมัส ลาตัส",
    "เพลี้ยแป้ง Pseudococcus comstocki", "พาราเธรนี รีแกลิส", "แอมเพโลฟากา", "เพลี้ยจักจั่นลายจุด Lycorma delicatula",
    "ด้วงหนวดยาว Xylotrechus", "จักจั่นเขียว Cicadella viridis", "แมลงมวนวงศ์ Miridae", "แมลงหวี่ขาวเรือนกระจก",
    "จักจั่นใบองุ่น Erythroneura apicalis", "ผีเสื้อหางติ่ง Papilio xuthus", "ไรส้ม Panonchus citri", "ไรสนิมส้ม Phyllocoptes oleiverus",
    "เพลี้ยหอยออสเตรเลีย", "เพลี้ยหอยยานอเนนซิส", "เพลี้ยหอยขี้ผึ้งแดง", "เพลี้ยหอยเกล็ดแดง",
    "เพลี้ยหอยพาร์ลาโทเรีย", "เพลี้ยแป้ง Nipaecoccus vastalor", "แมลงหวี่ขาวหนาม", "แมลงวันผลไม้ Bactrocera minax",
    "แมลงวันผลไม้ตะวันออก", "แมลงวันผลไม้สึเนโอนิส", "หนอนกระทู้ยาสูบ", "อะดริสไทรันนัส",
    "หนอนชอนใบส้ม", "เพลี้ยอ่อนดำส้ม", "เพลี้ยอ่อนเหลืองส้ม", "เพลี้ยอ่อนส้ม Aphis citricola",
    "เพลี้ยไฟพริก", "แมลงบั่ว Dasineura sp", "ลาวานา อิมิทาทา", "ซาลูร์นิส มาร์จิเนลลา",
    "ดีพอรัส มาร์จิเนทัส", "คลูมีเทีย ทรานส์เวอร์ซา", "เพลี้ยจักจั่นปากแบนมะม่วง", "ไรทิโดเดอรา โบวรินี",
    "ด้วงงวงมะม่วง Sternochetus frigidus", "จักจั่นวงศ์ Cicadellidae",
]

INSECT_LABELS = {
    idx: f"{th_name} ({en_name})"
    for idx, (th_name, en_name) in enumerate(zip(INSECT_LABELS_TH, INSECT_LABELS_EN))
}

if len(INSECT_LABELS_TH) != len(INSECT_LABELS_EN):
    st.warning("จำนวนชื่อไทยและอังกฤษไม่เท่ากัน ระบบจะแสดงผลเท่าที่จับคู่ได้")


def get_control_advice(english_name: str) -> str:
    name = english_name.lower()
    advice_by_group = [
        (["aphid", "plant hopper", "leafhopper", "cicadell", "miridae", "green bug", "toxoptera"],
         "สารแนะนำหลัก: pymetrozine, flonicamid | สารทางเลือก: imidacloprid, thiamethoxam"),
        (["thrips", "phloeothrips", "odontothrips", "scirtothrips"],
         "สารแนะนำหลัก: spinetoram, spinosad | สารทางเลือก: abamectin, emamectin benzoate"),
        (["borer", "stem", "leaf roller", "leaf caterpillar", "budworm", "citrella"],
         "สารแนะนำหลัก: chlorantraniliprole, emamectin benzoate | สารทางเลือก: indoxacarb, spinetoram"),
        (["army worm", "cutworm", "prodenia", "worm", "grub", "locustoidea"],
         "สารแนะนำหลัก: chlorantraniliprole, emamectin benzoate | สารทางเลือก: lambda-cyhalothrin, indoxacarb"),
        (["mite", "red spider", "panonchus", "colomerus", "phyllocoptes", "brevipoalpus", "penthaleus"],
         "สารแนะนำหลัก: spiromesifen, fenpyroximate | สารทางเลือก: abamectin, propargite"),
        (["weevil", "beetle", "xylotrechus", "sternochetus", "wireworm"],
         "สารแนะนำหลัก: fipronil, clothianidin | สารทางเลือก: bifenthrin, lambda-cyhalothrin"),
        (["whitefly", "trialeurodes", "aleurocanthus"],
         "สารแนะนำหลัก: buprofezin, pyriproxyfen | สารทางเลือก: spiromesifen, dinotefuran"),
        (["scale", "ceroplastes", "chrysomphalus", "parlatoria", "unaspis", "icerya"],
         "สารแนะนำหลัก: dinotefuran, imidacloprid | สารเสริมร่วม: horticultural oil/white oil"),
        (["fly", "midge", "dacus", "bactrocera", "dasineura"],
         "สารแนะนำหลัก: spinosad bait, malathion bait | สารทางเลือก: deltamethrin (พ่นเฉพาะจุดระบาด)"),
    ]

    for keywords, advice in advice_by_group:
        if any(keyword in name for keyword in keywords):
            return advice

    return "สารแนะนำทั่วไป: เลือกสารที่ขึ้นทะเบียนกับพืชเป้าหมาย และสลับกลุ่มสารออกฤทธิ์ทุก 1-2 รอบพ่น"


def get_damage_symptoms(english_name: str) -> str:
    name = english_name.lower()
    symptom_by_group = [
        (
            ["aphid", "plant hopper", "leafhopper", "cicadell", "miridae", "green bug", "toxoptera"],
            "ใบหงิกงอ เหลืองซีด ต้นแคระแกร็น และมีมูลหวาน/ราดำบนผิวใบ",
        ),
        (
            ["thrips", "phloeothrips", "odontothrips", "scirtothrips"],
            "ใบเกิดรอยเงินหรือรอยไหม้ ปลายใบม้วน แผลเป็นจุดถี่ และดอก/ยอดเสียรูป",
        ),
        (
            ["borer", "stem", "leaf roller", "leaf caterpillar", "budworm", "citrella"],
            "ใบถูกกัดกินหรือม้วนเป็นหลอด ลำต้น/ยอดถูกเจาะ มีมูลหนอน และต้นเหี่ยวเฉาเฉพาะส่วน",
        ),
        (
            ["army worm", "cutworm", "prodenia", "worm", "grub", "locustoidea"],
            "ใบแหว่งเป็นหย่อม ถูกกัดโคนต้นช่วงกลางคืน ต้นกล้าขาดล้ม และความเสียหายลามเร็ว",
        ),
        (
            ["mite", "red spider", "panonchus", "colomerus", "phyllocoptes", "brevipoalpus", "penthaleus"],
            "ใบเป็นจุดเหลืองถี่ ผิวใบกร้านหรือเป็นสนิม ใบแห้งกรอบ และมีใยบางในระยะระบาดหนัก",
        ),
        (
            ["weevil", "beetle", "xylotrechus", "sternochetus", "wireworm"],
            "รอยกัดกินบนใบ/ผล มีรูเจาะหรือตำหนิบนผล และชิ้นส่วนพืชเสียหายเป็นหย่อม",
        ),
        (
            ["whitefly", "trialeurodes", "aleurocanthus"],
            "ใบเหลืองซีด การสังเคราะห์แสงลดลง มีมูลหวานและราดำ ทำให้ผลผลิตลดคุณภาพ",
        ),
        (
            ["scale", "ceroplastes", "chrysomphalus", "parlatoria", "unaspis", "icerya"],
            "พบคราบเกาะกิ่ง/ใบ พืชทรุดโทรม ใบเหลืองร่วง และมีราดำตามมูลหวาน",
        ),
        (
            ["fly", "midge", "dacus", "bactrocera", "dasineura"],
            "ผลหรือยอดมีรอยวางไข่ เนื้อพืชช้ำ/เน่า ร่วงก่อนกำหนด และพบหนอนภายใน",
        ),
    ]

    for keywords, symptom in symptom_by_group:
        if any(keyword in name for keyword in keywords):
            return symptom

    return "สังเกตใบเหลืองหรือแผลผิดปกติ การเจริญเติบโตชะงัก และรอยกัดกิน/รอยเจาะตามส่วนพืช"


DEFAULT_API_URL = "https://mhrt03-insect-detector-demo.hf.space"


def _resolve_api_url() -> str:
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url

    # Streamlit raises if secrets.toml doesn't exist, so guard access.
    try:
        secret_url = st.secrets.get("API_URL")
        if secret_url:
            return str(secret_url)
    except Exception:
        pass

    return DEFAULT_API_URL


API_URL = _resolve_api_url()

st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-badge">เครื่องมือจำแนกภาพแมลง</div>
        <div class="hero-title">Pest InSight Classifier</div>
        <p class="hero-subtitle">อัปโหลดภาพเพื่อระบุชนิดแมลง พร้อมคำแนะนำป้องกันกำจัดเบื้องต้น</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="uploader-caption">อัปโหลดรูปแมลงได้หลายไฟล์ (JPG, JPEG, PNG) แล้วกดปุ่มจำแนกภาพ</p>', unsafe_allow_html=True)

with st.container():
    col_up, col_btn = st.columns([4, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "อัปโหลดรูปภาพแมลง",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    with col_btn:
        run_inference = st.button("จำแนกภาพ", type="primary", use_container_width=True)


if uploaded_files and run_inference:
    def predict_image(file):
        files = {"file": (file.name, file.getvalue(), file.type)}
        try:
            res = requests.post(f"{API_URL}/predict", files=files, timeout=60)
            if res.status_code == 200:
                return {"file": file, "data": res.json(), "error": None}
            return {"file": file, "data": None, "error": f"Error {res.status_code}"}
        except Exception as e:
            return {"file": file, "data": None, "error": str(e)}

    with st.spinner("AI กำลังจำแนกชนิดแมลง..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(predict_image, uploaded_files))

    st.divider()
    st.markdown(
        f"""
        <section class="result-header">
            <h2 class="result-title">ผลการจำแนกชนิดแมลง</h2>
            <div class="result-count">ทั้งหมด {len(results)} ภาพ</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cols_per_row = 4
    for i in range(0, len(results), cols_per_row):
        grid_cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(results):
                item = results[idx]
                with grid_cols[j]:
                    file_bytes = item["file"].getvalue()
                    mime_type = item["file"].type or "image/jpeg"
                    data_url = f"data:{mime_type};base64,{base64.b64encode(file_bytes).decode('utf-8')}"

                    if item["data"]:
                        p_id = item["data"]["prediction_class"]
                        conf = item["data"]["confidence"]
                        en_name = INSECT_LABELS_EN[p_id] if 0 <= p_id < len(INSECT_LABELS_EN) else "unknown pest"
                        display_name = INSECT_LABELS.get(p_id, f"Unknown (ID: {p_id})")
                        safe_name = escape(display_name)
                        safe_advice = escape(get_control_advice(en_name))
                        safe_symptom = escape(get_damage_symptoms(en_name))
                        progress_width = max(0.0, min(conf * 100, 100.0))

                        card_html = f"""
                        <div class="result-card">
                            <img class="result-image" src="{data_url}" alt="{safe_name}" />
                            <div class="card-info">
                                <span class="species-label">🐞 {safe_name}</span>
                                <div class="conf-text">ความมั่นใจของโมเดล: {conf*100:.2f}%</div>
                                <div class="conf-track"><div class="conf-fill" style="width:{progress_width:.2f}%"></div></div>
                                <span class="symptom-title">อาการทำลายที่พบบ่อย</span>
                                <div class="symptom-text">{safe_symptom}</div>
                                <span class="advice-title">สารเคมีแนะนำ (สารออกฤทธิ์)</span>
                                <div class="advice-text">{safe_advice}</div>
                                <div class="advice-note">หมายเหตุสำคัญ: ใช้เมื่อความเสียหายเกินระดับรุนแรงเท่านั้น และต้องตรวจฉลาก อัตราใช้ PHI REI รวมถึงสลับกลุ่มสาร (IRAC) ทุกครั้ง</div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        safe_error = escape(item["error"] or "Unknown error")
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <img class="result-image" src="{data_url}" alt="Prediction error" />
                                <div class="card-info"><span class="error-text">❌ {safe_error}</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
elif not uploaded_files:
    st.markdown(
        """
        <div class="empty-state">
            <strong>พร้อมเริ่มจำแนกภาพ</strong><br/>
            เลือกรูปภาพแมลง แล้วกดปุ่ม "จำแนกภาพ" เพื่อดูชนิดแมลง อาการทำลาย และสารเคมีแนะนำ
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<footer class="footer-note">AI Insect Classification for Field Use</footer>', unsafe_allow_html=True)