import argparse
from pathlib import Path


def create_sentence_pairs():
    time_phrases = [
        ("today", "hoy"),
        ("every morning", "cada mañana"),
        ("this afternoon", "esta tarde"),
        ("at night", "por la noche"),
        ("every day", "cada día"),
    ]
    subjects = [
        ("i", "yo", 0),
        ("you", "tú", 1),
        ("he", "él", 2),
        ("she", "ella", 3),
        ("we", "nosotros", 4),
        ("they", "ellos", 5),
    ]
    verbs = {
        "see": ["veo", "ves", "ve", "ve", "vemos", "ven"],
        "visit": ["visito", "visitas", "visita", "visita", "visitamos", "visitan"],
        "read": ["leo", "lees", "lee", "lee", "leemos", "leen"],
        "write": ["escribo", "escribes", "escribe", "escribe", "escribimos", "escriben"],
        "carry": ["llevo", "llevas", "lleva", "lleva", "llevamos", "llevan"],
        "open": ["abro", "abres", "abre", "abre", "abrimos", "abren"],
        "close": ["cierro", "cierras", "cierra", "cierra", "cerramos", "cierran"],
        "find": ["encuentro", "encuentras", "encuentra", "encuentra", "encontramos", "encuentran"],
        "need": ["necesito", "necesitas", "necesita", "necesita", "necesitamos", "necesitan"],
        "want": ["quiero", "quieres", "quiere", "quiere", "queremos", "quieren"],
        "hear": ["escucho", "escuchas", "escucha", "escucha", "escuchamos", "escuchan"],
        "bring": ["traigo", "traes", "trae", "trae", "traemos", "traen"],
    }
    objects = [
        ("the river", "el río"),
        ("the small house", "la casa pequeña"),
        ("the old bridge", "el puente viejo"),
        ("the garden", "el jardín"),
        ("the market", "el mercado"),
        ("the blue door", "la puerta azul"),
        ("the book", "el libro"),
        ("the teacher", "la maestra"),
        ("the flowers", "las flores"),
        ("the birds", "los pájaros"),
        ("the village", "el pueblo"),
        ("the warm bread", "el pan caliente"),
        ("the train station", "la estación de tren"),
        ("the new lantern", "la lámpara nueva"),
        ("the basket of apples", "la cesta de manzanas"),
        ("the school map", "el mapa de la escuela"),
    ]
    sentence_pairs = []

    for english_time_phrase, spanish_time_phrase in time_phrases:
        for english_subject, spanish_subject, subject_index in subjects:
            for english_verb, spanish_verb_forms in verbs.items():
                spanish_verb = spanish_verb_forms[subject_index]
                for english_object, spanish_object in objects:
                    english_sentence = (
                        f"{english_time_phrase} {english_subject} "
                        f"{english_verb} {english_object}."
                    )
                    spanish_sentence = (
                        f"{spanish_time_phrase} {spanish_subject} "
                        f"{spanish_verb} {spanish_object}."
                    )
                    sentence_pairs.append((english_sentence, spanish_sentence))

    return sentence_pairs


def main():
    application_directory = Path(__file__).parent
    argument_parser = argparse.ArgumentParser(
        description="Build a larger synthetic English-Spanish parallel corpus."
    )
    argument_parser.add_argument(
        "--seed-corpus-path",
        type=Path,
        default=application_directory / "seed_parallel_corpus.tsv",
    )
    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=application_directory / "parallel_corpus.tsv",
    )
    arguments = argument_parser.parse_args()

    seed_pairs = []
    for line_number, corpus_line in enumerate(
        arguments.seed_corpus_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not corpus_line.strip():
            continue
        sentence_pair = corpus_line.split("\t")
        if len(sentence_pair) != 2:
            argument_parser.error(f"seed corpus line {line_number} must contain one tab")
        seed_pairs.append((sentence_pair[0].strip(), sentence_pair[1].strip()))

    unique_sentence_pairs = list(dict.fromkeys(seed_pairs + create_sentence_pairs()))
    parallel_corpus_text = "".join(
        f"{english_sentence}\t{spanish_sentence}\n"
        for english_sentence, spanish_sentence in unique_sentence_pairs
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(parallel_corpus_text, encoding="utf-8")
    print(
        f"Wrote {len(unique_sentence_pairs)} English-Spanish sentence pairs "
        f"to {arguments.output_path}."
    )


if __name__ == "__main__":
    main()
