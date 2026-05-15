import logging

from datasets import load_dataset
import pandas as pd

from config import DATASET_NAME, LANGUAGES

logger = logging.getLogger(__name__)


def _transform_df(
    df: pd.DataFrame,
    languages: list = None,
    sample_per_class_per_lang: int = None,
) -> pd.DataFrame:
    if languages is None:
        languages = LANGUAGES

    required = {"language_name", "target", "code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset incompleto — colunas ausentes: {missing}")

    before = len(df)
    df = df[df["language_name"].isin(languages)].copy()
    logger.debug(
        "Filtro de linguagens %s aplicado: %d → %d amostras.",
        languages, before, len(df),
    )

    df["label"] = df["target"].apply(lambda x: "ai" if x == "Ai_generated" else "human")

    before = len(df)
    df = df.dropna(subset=["code"])
    df = df[df["code"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        logger.warning("%d amostra(s) removida(s) por código ausente ou vazio.", dropped)

    if sample_per_class_per_lang:
        logger.info(
            "Amostragem ativada: até %d exemplos por classe por linguagem.",
            sample_per_class_per_lang,
        )
        sampled = []
        for lang in languages:
            for label in ["ai", "human"]:
                subset = df[(df["language_name"] == lang) & (df["label"] == label)]
                n = min(sample_per_class_per_lang, len(subset))
                subset = subset.sample(n=n, random_state=42)
                logger.debug("  %-6s / %-5s : %d amostras selecionadas.", lang, label, n)
                sampled.append(subset)
        df = pd.concat(sampled).reset_index(drop=True)

    logger.info("Dataset pronto: %d amostras.", len(df))
    for (lang, label), count in df.groupby(["language_name", "label"]).size().items():
        logger.debug("  %-6s / %-5s : %d", lang, label, count)

    return df


def load_data(
    sample_per_class_per_lang: int = None,
    dataset_name: str = None,
    languages: list = None,
) -> pd.DataFrame:
    name = dataset_name or DATASET_NAME
    logger.info("Carregando dataset %s do HuggingFace...", name)
    ds = load_dataset(name, split="train")
    df = ds.to_pandas()
    logger.info("Dataset bruto carregado: %d amostras totais.", len(df))
    return _transform_df(df, languages=languages, sample_per_class_per_lang=sample_per_class_per_lang)
