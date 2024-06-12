import numpy as np
from matplotlib import pyplot as plt

# markov chain to compute if a culture of bacteriums contains a mutation after several generations to cure a virus

def main():

    pop = np.array([1, 0])
    m = np.array([[0.42, 0.58], [0.026, 0.974]])
    gen = np.arange(0, 18, dtype=int)
    popend = np.zeros((len(gen), 2))
    for i in range(len(gen)):
        if i == 0:
            popend[i, :] = pop
        else:
            popend[i, :] = popend[i-1, :] @ m

    # Definition: [1, 0] is first Generation
    ax = plt.figure().gca()
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.plot(gen+1, popend[:, 0], markevery=17, marker='o')
    plt.annotate(f'{popend[17, 0]*100:.2f} %', (17, popend[17, 0]))
    plt.xlabel('Generation')
    plt.ylabel('Probability of containing Mutation')
    plt.title(f'Chance of containing the mutation after {len(gen)} generations')
    plt.show()

    print(f'Probability of containing a mutation after 18 Generations: {popend[17, 0]*100:.2f} %')





if __name__ == '__main__':
    main()
