from langchain_core.prompts import PromptTemplate

# formatted string to use the input from user
usr_prompt = PromptTemplate(
    template = """
        Generate a detailed player profile for {player_name}.
        Include information such as career statistics, achievements, and notable performances.
            """,
    input_variables=["player_name"]
)

#for use of the same prompt in other scripts
usr_prompt.save("Player_Profile/player_name_prompt.json")