conda activate paprika

# AGENT=llama-3.1-8b-instruct
AGENT=qwen2.5-7B-Instruct

MODEL_MAX_LENGTH=32000

# Pick whichever task to run inference on
# Can have values between 0 and 9; set to -1 to run ALL environments
# If TASK_NUM is provided from the environment (e.g., by SLURM array), respect it
TASK_NUM="${TASK_NUM:--1}"

# Name of the task groups
GAME_ENVS=(
    "wordle"
    "cellular_automata"
    "mastermind"
    "battleship"
    "minesweeper"
    "bandit_bai_fixed_budget"
    "twenty_questions"
    "guess_my_city"
    "murder_mystery"
    "customer_service"
)

# Which task environment implementation to use
ENVS=(
    "wordle"
    "cellular_automata"
    "mastermind"
    "battleship"
    "minesweeper"
    "bandit_bai_fixed_budget"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
)

# Which task judge implementation to use
JUDGES=(
    "wordle"
    "cellular_automata"
    "mastermind"
    "battleship"
    "minesweeper"
    "bandit_bai_fixed_budget"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
)

if [ "$TASK_NUM" -ge 0 ]; then
    GAME_ENV=${GAME_ENVS[${TASK_NUM}]}
    ENV=${ENVS[${TASK_NUM}]}
    JUDGE=${JUDGES[${TASK_NUM}]}
fi


# which split of the data to use
# DATA_TYPE="train"
DATA_TYPE="eval"

# This allows one to run evaluation only on tasks 
# belonging to indices [START_INDEX, ...., END_INDEX - 1] within the task group.
# Mostly used to split evaluation across different GPUs, feel free to modify as required.
START_INDEX=0
END_INDEX=-1


# NOTE: In case one wants to run evaluation on the regular instruct model
# AGENT_MODEL_NAME="/scratch/gpfs/zs7353/Llama-3.1-8B-Instruct"
# FINETUNED_TOKENIZER=false
# AGENT_SAVE_FILE_NAME="Llama-3.1-8b-instruct"

# NOTE: Expect AGENT_MODEL_NAME to be provided via environment (e.g., by SLURM script)
: "${AGENT_MODEL_NAME:?AGENT_MODEL_NAME env var must be set (e.g., in SLURM script)}"
FINETUNED_TOKENIZER=false
# Sanitize model name for filesystem paths: replace all '/' with '_'
AGENT_SAVE_FILE_NAME="${AGENT_MODEL_NAME//\//_}"


# Temperature used for sampling from the agent
AGENT_TEMPERATURE=0.7
SEED=69   

# Num trajectories per task (e.g., a single secret topic in 20 questions) to be generated
NUM_TRAJECTORIES_PER_GAME_SCENARIO=4

# Modify the path here to reflect where the paprika directory is located
cd /home/zs7353/LBAT/paprika

run_one() {
    local idx=$1
    local game_env=${GAME_ENVS[$idx]}
    local env_impl=${ENVS[$idx]}
    local judge_impl=${JUDGES[$idx]}

    echo "Running task index $idx -> game_env=$game_env env=$env_impl judge=$judge_impl"
    python scripts/games/run_data_generation_game.py \
        seed=$SEED \
        agent=$AGENT \
        env=$env_impl \
        judge=$judge_impl \
        game_env=$game_env \
        game_env.data_type=$DATA_TYPE \
        start_index=$START_INDEX \
        end_index=$END_INDEX \
        agent.model_name=$AGENT_MODEL_NAME \
        agent.tokenizer_name=$AGENT_MODEL_NAME \
        agent.finetuned_tokenizer=$FINETUNED_TOKENIZER \
        agent.save_file_name=$AGENT_SAVE_FILE_NAME \
        agent.model_max_length=$MODEL_MAX_LENGTH \
        agent_temperature=$AGENT_TEMPERATURE \
        num_trajectories_per_game_scenario=$NUM_TRAJECTORIES_PER_GAME_SCENARIO \
        agent_model_supports_system_message=true
}

if [ "$TASK_NUM" -ge 0 ]; then
    run_one $TASK_NUM
else
    for i in "${!GAME_ENVS[@]}"; do
        run_one $i
    done
fi