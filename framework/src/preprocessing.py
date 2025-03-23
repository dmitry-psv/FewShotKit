from typing import Optional, Dict, Any, Literal
from textblob import TextBlob
from nltk.corpus import stopwords
import pandas as pd
import emoji
import re

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}

stopword = stopwords.words('english')

def basic_clean(text: str) -> str:
    """
    Базовая очистка текста 
    Args:
        text: Исходный текст
    Returns:
        Очищенный текст
    """
    text = text.strip()
    text = text.lower()
    return text

def remove_html_tags(text: str) -> str:
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_url(text: str) -> str:
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def chat_conversion(text: str) -> str:
    new_text = []
    for i in text.split():
        if i.upper() in chat_words:
            new_text.append(chat_words[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)

def remove_stopwords(text: str) -> str:
    new_text = []
    
    for word in text.split():
        if word in stopword:
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)

def spelling_correction(text: str) -> str:
    return str(TextBlob(text).correct().string)

def rewrite_emoji(text: str) -> str:
    return str(emoji.demojize(text))

def preprocess(text: str, functions: list) -> str:
    """
    Process single text using list of functions
    Args:
        text: Input text
        functions: List of preprocessing functions
    Returns:
        Processed text
    """
    for func in functions:
        text = func(text)
    return text

def process_dataset(df: pd.DataFrame, 
                   column: str, 
                   functions: list,
                   batch_size: int = 1000) -> pd.DataFrame:
    """
    Process entire dataset
    Args:
        df: Input dataframe
        column: Column name to process
        functions: List of preprocessing functions
        batch_size: Batch size for processing
    Returns:
        Processed dataframe
    """
    # Process text in batches
    texts = df[column].tolist()
    processed_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        processed_texts.extend([preprocess(text, functions) for text in batch])
    
    df[column] = processed_texts
    return df
