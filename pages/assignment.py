import csv
import streamlit as st
import random
import pandas as pd

# Title for the app
st.title("Genetic Algorithm Parameter Input")

# Inputs for crossover rate and mutation rate
CO_R = st.number_input(
    "Enter Crossover Rate (CO_R)", 
    min_value=0.0, max_value=0.95, step=0.01, value=0.8
)
MUT_R = st.number_input(
    "Enter Mutation Rate (MUT_R)", 
    min_value=0.01, max_value=0.20, step=0.01, value=0.2
)

# Display selected parameters
st.write("### Selected Parameters:")
st.write(f"- Crossover Rate (CO_R): {CO_R}")
st.write(f"- Mutation Rate (MUT_R): {MUT_R}")

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)
        
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    
    return program_ratings
    
# Path to the CSV file
file_path = 'pages/program_ratings.csv'
# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Print the result (you can also return or process it further)
for program, ratings in program_ratings_dict.items():
    st.write(f"'{program}': {ratings},")


import random

## DEFINING PARAMETERS AND DATASET 
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

GEN = 100
POP = 50
EL_S = 2

# Default values from the image instructions
default_co_r = 0.8
default_mut_r = 0.2

# Streamlit sliders for user input
co_r = st.sidebar.slider(
    "Crossover Rate (CO_R)", 
    min_value=0.0, 
    max_value=0.95, 
    value=default_co_r, 
    step=0.01
)

mut_r = st.sidebar.slider(
    "Mutation Rate (MUT_R)", 
    min_value=0.01, 
    max_value=0.05, 
    value=default_mut_r, 
    step=0.01
)

# Display the selected values
st.write("### Selected Parameters:")
st.write(f"- Crossover Rate (CO_R): {co_r}")
st.write(f"- Mutation Rate (MUT_R): {mut_r}")

all_programs = list(ratings.keys()) # all programs
all_time_slots = list(range(6, 24)) # time slots

### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# selection
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# calling the pop func.
all_possible_schedules = initialize_pop(all_programs, all_time_slots)

# callin the schedule func.
best_schedule = finding_best_schedule(all_possible_schedules)

############################################ GENETIC ALGORITHM #############################################################################

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# mutating
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# calling the fitness func.
def evaluate_fitness(schedule):
    return fitness_function(schedule)

# genetic algorithms with parameters
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):

    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitsm
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

# Function to create a schedule table
def create_schedule_table(schedule, time_slots):
    """
    Convert the schedule and time slots into a table format.
    Handles mismatched lengths by truncating or padding with placeholders.
    """
    # Ensure both lists are of the same length
    if len(schedule) < len(time_slots):
        schedule += ["No Program"] * (len(time_slots) - len(schedule))
    elif len(schedule) > len(time_slots):
        schedule = schedule[:len(time_slots)]
    
    data = {
        "Time Slot": [f"{slot}:00" for slot in time_slots],
        "Program": schedule
    }
    return pd.DataFrame(data)


##################################################### RESULTS ###################################################################################

# brute force
# Get the initial schedule from your brute-force or predefined logic
initial_best_schedule = finding_best_schedule(all_possible_schedules)

# Call the genetic algorithm with user inputs
final_schedule = genetic_algorithm(
    initial_schedule=initial_best_schedule,
    crossover_rate=CO_R,
    mutation_rate=MUT_R
)

# Display the results
st.write("### Final Optimal Schedule:")
for time_slot, program in enumerate(final_schedule):
    st.write(f"Time Slot {all_time_slots[time_slot]:02d}:00 - Program {program}")

st.write("Total Ratings:", fitness_function(final_schedule))

# Display the results
st.write("### Final Optimal Schedule:")
for time_slot, program in enumerate(final_schedule):
    st.write(f"Time Slot {all_time_slots[time_slot]:02d}:00 - Program {program}")

st.write("Total Ratings:", fitness_function(final_schedule))

# Create the table from the final schedule
schedule_table = create_schedule_table(final_schedule, all_time_slots)

# Display the table in Streamlit
st.write("### Schedule Table:")
st.table(schedule_table)  # Static table
# st.dataframe(schedule_table)  # Uncomment for interactive table
