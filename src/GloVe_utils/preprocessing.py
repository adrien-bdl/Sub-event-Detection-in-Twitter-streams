import pandas as pd
import re

### Class and functions for the preprocessing


class Patterns:
    URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
    HASHTAG_PATTERN = re.compile("#")
    MENTION_PATTERN = re.compile(r"@\w+")
    # RT_PATTERN = re.compile(r'\bRT :\b', re.IGNORECASE)  # Case-insensitive pattern
    RT_PATTERN = re.compile("RT :")
    EMOJIS_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
    SMILEYS_PATTERN = re.compile(r"[:;=8][\-o\*\']?[\)\(\]dDpP/\:\}\{@\|\\]|<3")
    NUMBERS_PATTERN = re.compile(r"\b\d+\b")
    CHARACTER_NORMALIZATION_PATTERN = re.compile(
        r"([A-Za-z])\1{2,}|([!?.])\2{1,}"
    )  # remove the iteration when a letter appears more that 2 times in a row ex: goalllllll -> goal and punctuation characters repeated more than once
    QUOTES_PATTERN = re.compile('"')


class Defines:
    FILTERED_METHODS = [
        "newlines",
        "urls",
        "mentions",
        "hashtags",
        "rt",
        "emojis",
        "smileys",
        "numbers",
        "escape_chars",
        "lower",
        "character_normalization",
        "quotes",
    ]  # prioritised order
    PREPROCESS_METHODS_PREFIX = "preprocess_"


opts = {
    "NEWLINES": "newlines",
    "URL": "urls",
    "MENTION": "mentions",
    "HASHTAG": "hashtags",
    "RT": "rt",
    "EMOJI": "emojis",
    "SMILEY": "smileys",
    "NUMBER": "numbers",
    "ESCAPE_CHAR": "escape_chars",
    "Lower": "lower",
    "CHARACTER_NORMALIZATION": "character_normalization",
    "QUOTES": "quotes",
}


class TweetCleaner:
    def get_worker_methods(self, prefix):

        # Filtering according to user's options
        prefixed_filtered_methods = [prefix + fm for fm in Defines.FILTERED_METHODS]

        return prefixed_filtered_methods

    def clean(self, tweet_string):
        repl = "clean"
        cleaner_methods = self.get_worker_methods(Defines.PREPROCESS_METHODS_PREFIX)
        for a_cleaner_method in cleaner_methods:
            method_to_call = getattr(self, a_cleaner_method)

            tweet_string = method_to_call(tweet_string, "")

        tweet_string = self.remove_unnecessary_characters(tweet_string)
        return tweet_string

    def preprocess_urls(self, tweet_string, repl):
        return Patterns.URL_PATTERN.sub(repl, tweet_string)

    def preprocess_hashtags(self, tweet_string, repl):
        return Patterns.HASHTAG_PATTERN.sub(repl, tweet_string)

    def preprocess_mentions(self, tweet_string, repl):
        return Patterns.MENTION_PATTERN.sub(repl, tweet_string)

    def preprocess_rt(self, tweet_string, repl):
        return Patterns.RT_PATTERN.sub(
            repl, tweet_string
        )  # Correct case-insensitive pattern for 'RT'

    def preprocess_reserved_words(self, tweet_string, repl):
        return Patterns.RESERVED_WORDS_PATTERN.sub(repl, tweet_string)

    def preprocess_emojis(self, tweet_string, repl):
        processed = Patterns.EMOJIS_PATTERN.sub(repl, tweet_string)
        return processed.encode("ascii", "ignore").decode("ascii")

    def preprocess_smileys(self, tweet_string, repl):
        return Patterns.SMILEYS_PATTERN.sub(repl, tweet_string)

    def preprocess_numbers(self, tweet_string, repl):
        return re.sub(
            Patterns.NUMBERS_PATTERN, lambda m: m.group(0) + repl, tweet_string
        )

    def preprocess_escape_chars(self, tweet_string, repl):
        """
        This method processes escape chars using ASCII control characters.
        :param tweet_string: input string which will be used to remove escape chars
        :param repl: unused for this method
        :return: processed string
        """
        escapes = "".join([chr(char) for char in range(1, 32)])
        translator = str.maketrans("", "", escapes)
        return tweet_string.translate(translator)

    def preprocess_lower(self, tweet_string, repl):
        return tweet_string.lower()  # Convert text to lowercase

    def preprocess_newlines(self, tweet_string, repl):
        return tweet_string.replace("\n", " ")

    def remove_unnecessary_characters(self, tweet_string):
        return " ".join(tweet_string.split())

    def preprocess_character_normalization(self, tweet_string, repl):
        return Patterns.CHARACTER_NORMALIZATION_PATTERN.sub(
            lambda m: m.group(1) if m.group(1) else m.group(2), tweet_string
        )

    def preprocess_quotes(self, tweet_string, repl):
        return Patterns.QUOTES_PATTERN.sub(repl, tweet_string)


def replace_names_in_tweet(tweet, filename):
    if not isinstance(tweet, str):  # Check if tweet is a string
        return tweet  # Return tweet as is if it's not a string (e.g., NaN or float)

    tweet_lower = tweet.lower()

    players_2010 = pd.read_csv("../players/2010_players.csv")
    players_2014 = pd.read_csv("../players/2014_players.csv")

    # Create dictionaries for players from both teams (2010 and 2014)
    # Ensure each player name is a string and convert it to lowercase
    team_1_players_2010 = {
        str(player).lower(): f"player_2010_{i+1}"
        for i, player in enumerate(players_2010.iloc[0, 1:])
        if isinstance(player, str)
    }
    team_2_players_2010 = {
        str(player).lower(): f"player_2010_{i+1}"
        for i, player in enumerate(players_2010.iloc[1, 1:])
        if isinstance(player, str)
    }

    team_1_players_2014 = {
        str(player).lower(): f"player_2014_{i+1}"
        for i, player in enumerate(players_2014.iloc[0, 1:])
        if isinstance(player, str)
    }
    team_2_players_2014 = {
        str(player).lower(): f"player_2014_{i+1}"
        for i, player in enumerate(players_2014.iloc[1, 1:])
        if isinstance(player, str)
    }

    # Replace player names from the 2010 squad
    for player in team_1_players_2010:
        if player in tweet_lower:
            tweet = tweet.replace(player, team_1_players_2010[player])

    for player in team_2_players_2010:
        if player in tweet_lower:
            tweet = tweet.replace(player, team_2_players_2010[player])

    # Replace player names from the 2014 squad
    for player in team_1_players_2014:
        if player in tweet_lower:  #
            tweet = tweet.replace(player, team_1_players_2014[player])

    for player in team_2_players_2014:
        if player in tweet_lower:
            tweet = tweet.replace(player, team_2_players_2014[player])

    return tweet
