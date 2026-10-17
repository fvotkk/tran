import argparse
import re
from pathlib import Path


def create_generated_chapters(chapter_count):
    seasons = ["spring", "summer", "autumn", "winter"]
    villages = [
        "Bellweather",
        "Willowmere",
        "Stonebridge",
        "Northfield",
        "Greenhollow",
        "Clearwater",
        "Meadowgate",
        "Ashford",
    ]
    characters = [
        "Mira",
        "Elian",
        "Nora",
        "Tomas",
        "Lena",
        "Rafael",
        "Iris",
        "Mateo",
        "Sofia",
        "Oren",
        "Clara",
        "Jules",
    ]
    landmarks = [
        "the river mill",
        "the eastern hill",
        "the old bridge",
        "the village garden",
        "the market square",
        "the pine woods",
        "the harbor path",
        "the schoolhouse",
        "the lighthouse",
        "the orchard gate",
    ]
    landscapes = [
        "open fields",
        "green hills",
        "a rocky shore",
        "a quiet valley",
        "a wide river",
        "a bright orchard",
        "a line of tall pines",
        "a patchwork of gardens",
    ]
    weather_conditions = [
        "a cool breeze",
        "soft rain",
        "warm sunlight",
        "low mist",
        "a steady wind",
        "gentle snow",
        "clear morning air",
        "a passing storm",
    ]
    carried_objects = [
        "a wooden toolbox",
        "a basket of apples",
        "a folded map",
        "a brass lantern",
        "a loaf of fresh bread",
        "a bundle of letters",
        "a blue wool blanket",
        "a small music box",
        "a sack of wheat",
        "a jar of honey",
        "a repaired clock",
        "a painted sign",
    ]
    shared_tasks = [
        "repair the footbridge",
        "prepare the market stalls",
        "find a safe path through the woods",
        "deliver medicine to the far bank",
        "restore the garden wall",
        "teach the children to read a map",
        "carry supplies to the hilltop shelter",
        "clear leaves from the mill channel",
        "return a lost parcel to its owner",
        "build a bench beside the river",
    ]
    discoveries = [
        "a narrow path hidden beneath the ivy",
        "a wooden box full of old letters",
        "fresh tracks beside the water",
        "a broken sign pointing toward the harbor",
        "a family of birds beneath the roof",
        "a spring bubbling under a stone wall",
        "a forgotten key beside the gate",
        "a basket that had fallen from a cart",
        "a lantern still glowing in the rain",
        "a set of drawings made by a village child",
    ]
    obstacles = [
        "the road had become muddy",
        "the bridge was blocked by fallen branches",
        "the wind kept pushing the door shut",
        "the river had risen above its usual mark",
        "the path disappeared beneath fresh snow",
        "the cart wheel had come loose",
        "the market clock had stopped",
        "the rain had covered the trail with leaves",
        "the lantern needed a new wick",
        "the last boat had already crossed the river",
    ]
    solutions = [
        "they asked the carpenter to bring a longer beam",
        "they marked the safe route with bright ribbons",
        "they shared their tools and worked in turns",
        "they waited for the water to fall before crossing",
        "they followed the tracks back to the village",
        "they repaired the wheel with a piece of oak",
        "they lit the path with lanterns from the school",
        "they asked each neighbor to carry one small bundle",
        "they read the old map together and found another way",
        "they built a shelter from boards stored in the mill",
    ]
    endings = [
        "everyone reached home before the evening bell",
        "the market opened on time the following morning",
        "the children added the discovery to their village notebook",
        "the traveler thanked the neighbors with a song",
        "the river path became safer for the next season",
        "the repaired place served the whole village for many years",
        "the neighbors gathered to share soup and warm bread",
        "the small problem became a story told at winter gatherings",
        "the village learned to prepare better for the next storm",
        "the work brought two distant families together",
    ]
    generated_chapters = []

    for chapter_index in range(chapter_count):
        character = characters[chapter_index % len(characters)]
        neighbor = characters[(chapter_index * 5 + 3) % len(characters)]
        village = villages[(chapter_index * 3) % len(villages)]
        season = seasons[(chapter_index * 7) % len(seasons)]
        landmark = landmarks[(chapter_index * 3 + 1) % len(landmarks)]
        landscape = landscapes[(chapter_index * 5 + 2) % len(landscapes)]
        weather = weather_conditions[(chapter_index * 7 + 4) % len(weather_conditions)]
        carried_object = carried_objects[(chapter_index * 5 + 1) % len(carried_objects)]
        shared_task = shared_tasks[(chapter_index * 7 + 2) % len(shared_tasks)]
        discovery = discoveries[(chapter_index * 3 + 5) % len(discoveries)]
        obstacle = obstacles[(chapter_index * 9 + 1) % len(obstacles)]
        solution = solutions[(chapter_index * 7 + 6) % len(solutions)]
        ending = endings[(chapter_index * 3 + 2) % len(endings)]

        generated_chapters.append(
            " ".join(
                [
                    f"Chapter {chapter_index + 1} began in {village} during {season}.",
                    f"Before sunrise, {character} walked toward {landmark} with {carried_object}.",
                    f"Across {landscape}, {weather} moved slowly toward the village.",
                    f"{neighbor} was already waiting there, ready to help {character} {shared_task}.",
                    f"On the way, they discovered {discovery}.",
                    f"They could not continue at once because {obstacle}.",
                    f"After listening to one another, {solution}.",
                    f"By evening, {ending}.",
                ]
            )
        )

    return generated_chapters


def main():
    application_directory = Path(__file__).parent
    argument_parser = argparse.ArgumentParser(
        description="Build a larger original story corpus for the word-level RNN."
    )
    argument_parser.add_argument("--chapter-count", type=int, default=240)
    argument_parser.add_argument(
        "--seed-story-path",
        type=Path,
        default=application_directory / "seed_story.txt",
    )
    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=application_directory / "corpus.txt",
    )
    arguments = argument_parser.parse_args()

    if arguments.chapter_count < 1:
        argument_parser.error("chapter count must be positive")

    seed_story = arguments.seed_story_path.read_text(encoding="utf-8").strip()
    generated_chapters = create_generated_chapters(arguments.chapter_count)
    corpus_text = f"{seed_story}\n\n" + "\n\n".join(generated_chapters) + "\n"
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(corpus_text, encoding="utf-8")
    token_count = len(re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", corpus_text.lower()))
    print(
        f"Wrote {token_count} word and punctuation tokens from "
        f"{arguments.chapter_count} generated chapters to {arguments.output_path}."
    )


if __name__ == "__main__":
    main()
