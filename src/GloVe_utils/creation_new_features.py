import re
from sklearn.preprocessing import MinMaxScaler


def add_tweet_features(df):

    regex_repeated_letters = r"(.)\1{2,}"
    df["Has_Repeated_Letters"] = df["Tweet"].str.contains(regex_repeated_letters)

    emoji_regex = r"[\U00010000-\U0010ffff]"
    df["Emoji_Count"] = df["Tweet"].apply(lambda x: len(re.findall(emoji_regex, x)))

    df["Exclamation_Count"] = df["Tweet"].apply(lambda x: x.count("!"))
    df["Question_Count"] = df["Tweet"].apply(lambda x: x.count("?"))

    df["starts_with_RT"] = df["Tweet"].str.startswith("RT")
    df["isMention"] = df["Tweet"].str.contains("@")

    agg_features = {
        "Has_Repeated_Letters": "sum",
        "Emoji_Count": "sum",
        "Exclamation_Count": "sum",
        "Question_Count": "sum",
        "starts_with_RT": "sum",
        "isMention": "sum",
    }

    df_agg = df.groupby("ID").agg(agg_features).reset_index()

    # Calculate features per minute
    df_agg["nb_tweets_per_minute"] = df.groupby("ID")["ID"].transform("count")
    df_agg["nb_consecutive_letters_per_minute"] = df_agg["Has_Repeated_Letters"]
    df_agg["nb_smileys_per_minute"] = df_agg["Emoji_Count"]
    df_agg["Exclamation_Count_per_minute"] = df_agg["Exclamation_Count"]
    df_agg["Question_Count_per_minute"] = df_agg["Question_Count"]
    df_agg["nb_RT_per_min"] = df_agg["starts_with_RT"]
    df_agg["nb_@_per_min"] = df_agg["isMention"]
    df_agg["Match_time"] = df_agg["ID"].str.split("_").str[1].astype(int)

    # Define features to normalize
    features_to_normalize = [
        "nb_tweets_per_minute",
        "nb_consecutive_letters_per_minute",
        "nb_smileys_per_minute",
        "Exclamation_Count_per_minute",
        "Question_Count_per_minute",
        "nb_RT_per_min",
        "nb_@_per_min",
    ]

    scaler = MinMaxScaler()
    df_normalized = df_agg.copy()
    df_normalized[features_to_normalize] = scaler.fit_transform(
        df_normalized[features_to_normalize]
    )
    df_new_features = df_normalized.drop_duplicates(subset="ID")

    return df_new_features
