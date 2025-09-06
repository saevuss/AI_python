import random
from deap import base, creator, tools

def eval_func(individual):
    ''' defining the evaluation funcion: first step to create a genetic algorithm '''
    target_sum = 15
    value = len(individual) - abs(sum(individual) - target_sum)
    return (value,) #the more the sum is near 15, the more the fitness'evaluation is high

def create_toolbox(num_bits):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    #inizializing the toolbox
    toolbox = base.Toolbox()
    #gene = bit 0 or 1
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_bits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # population = list of individuals

    #registering the evaluation operator
    toolbox.register("evaluate", eval_func)
    #register the corssover operator
    toolbox.register("mate", tools.cxTwoPoint)
    #register a mutation operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) #flip of bits with prob of 5%

    #define the operator for breeding
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

if __name__ == "__main__":
    ''' l'idea è creare una sequenza di 45 bit (0 o 1) la cui somma sia 15. si vuole copiare l'idea
    dell'evoluzione naturale'''
    num_bits = 45
    toolbox = create_toolbox(num_bits)
    random.seed(7)
    population = toolbox.population(n=500) #initial population of 500 individuals
    probab_crossing, probab_mutating = 0.5, 0.2
    num_generations = 10 #for 10 generation there are different steps
    print('\nEvolution process starts')

    #FIRST STEP: SELECTION --> most suitable individuals
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print('\nEvaluated', len(population), 'individuals')

    #EVOLUTIONARY CYCLE
    for g in range(num_generations):
        print('\nGeneration' , g)
        #selecting the next generation individuals
        offspring = toolbox.select(population, len(population)) #scambia pezzi tra individui (mescola i geni)
        #now clone the selected individuals (to not modify the original's ones)
        offspring = list(map(toolbox.clone, offspring))
        #applying crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)
                #delete the fintess value of child
                del child1.fitness.values
                del child2.fitness.values
        #now apply mutation
        for mutant in offspring: #cambia qualche bit a caso
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        #evaluate the individuals with an invalid fintess
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print('\nEvaluated', len(invalid_ind), 'individuals')
        #substituting the population with the new generation
        population[:] = offspring

        #STATISTIC OF THE GENERATION
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits)/length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2/length -mean**2)**0.5
        print('Min= ', min(fits), ', Max = ', max(fits))
        print('Average= ', round(mean, 2), ', Standard deviation = ', round(std, 2))
        print('\nEvaluation end')