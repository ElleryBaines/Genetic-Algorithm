# Ellery Baines 
import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# parameters, warning: fragile code, combinations not thoroughly tested
random_sampling = True # otherwise index order of sample
printouts = True # see stuff
save_data = True # save avg fitness scores by iteration output
half_children = True # either pop/2, or pop/4 in child generation size (half_children == True == pop/4)
save_title = 'avg_fitness_by_iteration1.txt' # path to save avg fitness by iteration output
path = 'C:\\Users\\Ellery\\Documents\\Masters\\Machine Learning\\Project\\SportsArticles\\features.xlsx' # path to load features dataset input
population_size = 20 # > 0 & multiple of 4
chromo_len = 10 # > 0 & divisible by 2
train_size_from_sample = .8 # between [0,1]
mutation_rate = .8 # betweeon [0,1]
iterations = 200 # > 0
crossover_perc = .5 # between [0,1] fragile, not fully tested, usually at .5

# list of sheet numbers and names, 0 is dataframes of features, 1 is verbose titles/details of the set
sheet = [0, 1] 

# copy dataset to variable
dataset = pd.read_excel(io=path, sheet_name=sheet)

# get sample size n, and set list of features to test, currently selected features match Model I
n = len(dataset[0])
features_to_test = [       
    #'TextID',
    #'URL',
    #'Label',
    #'totalWordsCount',
    #'semanticobjscore',
    #'semanticsubjscore',
    #'CC',
    #'CD',
    #'DT',
    #'EX',
    #'FW',
    #'INs',
    #'JJ',
    #'JJR',
    #'JJS',
    #'LS',
    #'MD',
    #'NN',
    #'NNP',
    #'NNPS',
    #'NNS',
    #'PDT',
    #'POS',
    #'PRP',
    #'PRP$',
    #'RB',
    #'RBR',
    #'RBS',
    #'RP',
    #'SYM',
    #'TOs',
    #'UH',
    #'VB',
    #'VBD',
    #'VBG',
    #'VBN',
    #'VBP',
    #'VBZ',
    #'WDT',
    #'WP',
    #'WP$',
    #'WRB',
    #'baseform',
    'Quotes',
    'questionmarks',
    'exclamationmarks',
    #'fullstops',
    #'commas',
    #'semicolon',
    #'colon',
    #'ellipsis',
    'pronouns1st',
    'pronouns2nd',
    'pronouns3rd',
    'compsupadjadv',
    'past',
    'imperative',
    'present3rd',
    'present1st2nd',
    #'sentence1st',
    'sentencelast',
    #'txtcomplexity'
]

# seperate data in train and test sets
# either random sample or index order of sample is choosen
test_set = pd.DataFrame(columns=dataset[0].columns.values)
train_set = pd.DataFrame(columns=dataset[0].columns.values)

if random_sampling:
    msk = random.sample(list(range(0, n)), int(n * train_size_from_sample)) 
    for x in range(0, n):
        if x in msk:
            train_set = train_set.append(dataset[0].iloc[x])
        else:
            test_set = test_set.append(dataset[0].iloc[x])
else:
    split = int(n * train_size_from_sample)
    train_set = dataset[0].iloc[0 : split]
    test_set = dataset[0].iloc[split : n]

# split test/train sets into features and labels
train_X = train_set[features_to_test]
test_X = test_set[features_to_test]

train_Y = train_set['Label']
test_Y = test_set['Label'] 

# Genetic Algorithm
chromosome_arr = []
chromosome_fitness = []
avg_fitness_by_iteration = []

# create weights set to 1
weights = {}
i_count = 0
for x in features_to_test:
    # [index that corresponds to chromosome, value]
    weights[x] = [i_count, 1]
    i_count = i_count + 1

# feature used in both sub & obj calc, thus separate weights
weights['present3rdobj'] = [i_count, 1]
weights['present1st2ndobj'] = [i_count + 1, 1]
weights['present3rdsubj'] = [i_count + 2, 1]
weights['present1st2ndsubj'] = [i_count + 3, 1]

# find min fitness of list of 2d arrays
def find_min_fitness_index(fitness_arr):
    min_index = fitness_arr[0][0]
    min_fitness = fitness_arr[0][1]
    for x in range(0, len(fitness_arr)):
        if(fitness_arr[x][1] < min_fitness):
            min_fitness = fitness_arr[x][1]
            min_index = fitness_arr[x][0]
    return min_index

# fitness testing function
def score_fitness(chromosome):
    weights_encoded = {}
    
    # dimensionality reduction of chromosome's feature (input 20) to single weight (input 1)
    pca = PCA(n_components=1)    
    principalComponents = pca.fit_transform(chromosome)

    # calibrate weights to chromosome
    for x in weights:
        weight_index = weights[x][0]
        weights_encoded[x] = weights[x][1] * principalComponents[weight_index][0]
    
    # metrics
    true_obj = 0
    false_obj = 0
    true_subj = 0
    false_subj = 0
    unknown_count = 0
    fitness = 0

    # summation of obj and subj features w/ weights
    for x in range(0, len(train_X)):
        # obj features, from Model I = 'Quotes', 'pronouns3rd', 'past', 'present3rd', 'present1st2nd'
        obj_sum = (train_X.iloc[x]['Quotes'] * weights_encoded['Quotes']) + \
            (train_X.iloc[x]['pronouns3rd'] * weights_encoded['pronouns3rd']) + \
                (train_X.iloc[x]['past'] * weights_encoded['past']) + \
                    (train_X.iloc[x]['present3rd'] * weights_encoded['present3rdobj']) + \
                        (train_X.iloc[x]['present1st2nd'] * weights_encoded['present1st2ndobj'])

        # subj features, from Model I = 'questionmarks', 'exclamationmarks', 'pronouns1st', 'pronouns2nd', 'imperative', 'present3rd', 'present1st2nd', 'compsupadjadv'    
        subj_sum = (train_X.iloc[x]['questionmarks'] * weights_encoded['questionmarks']) + \
            (train_X.iloc[x]['exclamationmarks'] * weights_encoded['exclamationmarks']) + \
                (train_X.iloc[x]['pronouns1st'] * weights_encoded['pronouns1st']) + \
                    (train_X.iloc[x]['pronouns2nd'] * weights_encoded['pronouns2nd']) + \
                        (train_X.iloc[x]['imperative'] * weights_encoded['imperative']) + \
                            (train_X.iloc[x]['present3rd'] * weights_encoded['present3rdsubj']) + \
                                (train_X.iloc[x]['present1st2nd'] * weights_encoded['present1st2ndsubj']) + \
                                    (train_X.iloc[x]['compsupadjadv'] * weights_encoded['compsupadjadv']) 

        # last sentence
        ls_score = (train_X.iloc[x]['sentencelast'] * weights_encoded['sentencelast'])

        # class score
        class_score = obj_sum - subj_sum + ls_score

        # calculate metrics     
        if(class_score >= 1):
            if(train_Y.iloc[x] == 'objective'):
                true_obj = true_obj + 1
            else:
                false_obj = false_obj + 1
        elif(class_score <= -1):
            if(train_Y.iloc[x] == 'subjective'):
                true_subj = true_subj + 1
            else:
                false_subj = false_subj + 1
        else:
            unknown_count = unknown_count + 1
    fitness = true_obj + true_subj
    if printouts:
        print(true_obj, 'true_obj')
        print(false_obj, 'false_obj')
        print(true_subj, 'true_subj')
        print(false_subj, 'false_subj')
        print(unknown_count, 'unknown_count')
        print(true_obj + false_obj + true_subj + false_subj + unknown_count, 'total')
        print(fitness, 'fitness of chromosome\n')
    return fitness

# starting population, generating random chromosomes
for x in range(0, population_size):
    chromosome = []
    for y in range(0, len(weights)):
        feature = []
        for z in range(0, chromo_len):
            feature.append(random.randint(0, 1))
        chromosome.append(feature)
    chromosome_arr.append(chromosome)

# score intial population
for i in range(0, population_size):
    if printouts:
        print('scoring population member ', i)
    chromosome_fitness.append([i, score_fitness(chromosome_arr[i])])

# main loop for iterations
for iteration in range(0, iterations):
    print('\n~ iteration : ', iteration, '~\n')
        
    # crossover canidate choice
    crossover_set = random.sample(list(range(0, population_size)), int(crossover_perc * population_size))
    children_arr = []

    for x in range(0, int(len(crossover_set)/2)):
        # Choose random 2 cut points
        cuts = [[], []]
        cuts[0].append(random.randint(0, len(chromosome_arr[0]) - 1))
        cuts[0].append(random.randint(0, chromo_len) - 1)
        cuts[1].append(random.randint(0, len(chromosome_arr[0]) - 1))
        cuts[1].append(random.randint(0, chromo_len) - 1)
        
        cut_order = np.argsort(cuts, axis=0)
        cuts_in_order = []

        for y in range(0, len(cuts)):
            cuts_in_order.append([])
        while(cuts[0][0] == cuts[1][0]):
            cuts[1][0] = random.randint(0, len(chromosome_arr[0]) - 1)
            cut_order = np.argsort(cuts, axis=0)
            cuts_in_order = []
            for y in range(0, len(cuts)):
                cuts_in_order.append([])
        for y in range(0, len(cuts)):
            cuts_in_order[y] = cuts[cut_order[y][0]]
        
        # make children using double crossover at cuts
        parent1 = crossover_set[x]
        parent2 = crossover_set[x + int(len(crossover_set)/2)]

        # child1 = parent1/parent2 single cross
        first_half = chromosome_arr[parent1][:cuts_in_order[0][0]]
        second_half = chromosome_arr[parent2][cuts_in_order[0][0]:]

        list1 = chromosome_arr[parent1][cuts_in_order[0][0]][:cuts_in_order[0][1]]
        list2 = chromosome_arr[parent2][cuts_in_order[0][0]][cuts_in_order[0][1]:]
        merge_list = []
        for y in list1:
            merge_list.append(y)
        for y in list2:
            merge_list.append(y)
        
        child1 = []
        for y in first_half:
            child1.append(y)
        for y in second_half:
            child1.append(y)
        child1[cuts_in_order[0][0]] = merge_list

        # child2 = parent2/parent1 single cross
        first_half = chromosome_arr[parent2][:cuts_in_order[0][0]]
        second_half = chromosome_arr[parent1][cuts_in_order[0][0]:]

        list1 = chromosome_arr[parent2][cuts_in_order[0][0]][:cuts_in_order[0][1]]
        list2 = chromosome_arr[parent1][cuts_in_order[0][0]][cuts_in_order[0][1]:]
        merge_list = []
        for y in list1:
            merge_list.append(y)
        for y in list2:
            merge_list.append(y)
        
        child2 = []
        for y in first_half:
            child2.append(y)
        for y in second_half:
            child2.append(y)
        child2[cuts_in_order[0][0]] = merge_list

        # child3 = child1/child2 single cross 
        first_half = child1[:cuts_in_order[1][0]]
        second_half = child2[cuts_in_order[1][0]:]

        list1 = child1[cuts_in_order[1][0]][:cuts_in_order[1][1]]
        list2 = child2[cuts_in_order[1][0]][cuts_in_order[1][1]:]
        merge_list = []
        for y in list1:
            merge_list.append(y)
        for y in list2:
            merge_list.append(y)
        
        child3 = []
        for y in first_half:
            child3.append(y)
        for y in second_half:
            child3.append(y)
        child3[cuts_in_order[1][0]] = merge_list

        # child4 = child2/child1 single cross 
        first_half = child2[:cuts_in_order[1][0]]
        second_half = child1[cuts_in_order[1][0]:]

        list1 = child2[cuts_in_order[1][0]][:cuts_in_order[1][1]]
        list2 = child1[cuts_in_order[1][0]][cuts_in_order[1][1]:]
        merge_list = []
        for y in list1:
            merge_list.append(y)
        for y in list2:
            merge_list.append(y)
        
        child4 = []
        for y in first_half:
            child4.append(y)
        for y in second_half:
            child4.append(y)
        child4[cuts_in_order[1][0]] = merge_list

        children_arr.append(child3)
        if(half_children == False):
            children_arr.append(child4)

    # single point mutatation in children
    for x in range(0, len(children_arr)):
        if(random.uniform(0, 1) <= mutation_rate):
            row = random.randint(0, len(chromosome_arr[0]) - 1)
            col = random.randint(0, chromo_len - 1)
            if(children_arr[x][row][col] == 1):
                children_arr[x][row][col] = 0
            else:
                children_arr[x][row][col] = 1

    # score fitness of children
    children_fitness_arr = []
    for x in range(0, len(children_arr)):
        if printouts:
            print('scoring child ', x)
        fitness = score_fitness(children_arr[x])
        children_fitness_arr.append([x, fitness])

    # replacement technique
    for x in range(0, len(children_fitness_arr)):
        min_index = find_min_fitness_index(chromosome_fitness)

        if(children_fitness_arr[x][1] > chromosome_fitness[min_index][1]):
            if printouts:
                print('swaping in child ', x, ' fitness = ', children_fitness_arr[x][1], ', swaping out population ', min_index, ' fitness = ', chromosome_fitness[min_index][1])
            chromosome_arr[min_index] = children_arr[x]
            chromosome_fitness[min_index][1] = children_fitness_arr[x][1]
            min_index = find_min_fitness_index(chromosome_fitness)  

    # record avg fitness, check if perfect fitness
    fitness_sum = 0
    for x in range(0, len(chromosome_fitness)):
        fitness_sum = fitness_sum + chromosome_fitness[x][1]
        # win condition
        if(chromosome_fitness[x][1] == len(train_X)):
            np.save(save_title, avg_fitness_by_iteration)
            np.savetxt(save_title, avg_fitness_by_iteration)
            with open("output.txt", "w") as txt_file:
                for line in avg_fitness_by_iteration:
                    txt_file.write(" ".join(str(line)) + "\n")
            print('\nperfect fitness score achieved ~ prophet or overfit?\n')
            break
    fitness_avg = fitness_sum/len(chromosome_fitness)

    if printouts:
        print('iteration average fitness : ', fitness_avg)
    avg_fitness_by_iteration.append([iteration, fitness_avg])

# save output, many times, because I worry
if(save_data):
    np.save(save_title, avg_fitness_by_iteration)
    np.savetxt(save_title, avg_fitness_by_iteration)
    with open("output1.txt", "w") as txt_file:
        for line in avg_fitness_by_iteration:
            txt_file.write(" ".join(str(line)) + "\n")
    with open("chromosomes1.txt", "w") as txt_file:
        for line in chromosome_arr:
            txt_file.write(" ".join(str(line)) + "\n")

# output
if printouts:
    print('\niteration, fitness_avg')
    for x in avg_fitness_by_iteration:
        print(x)   

    print('\nGoodbye Stranger...')
    