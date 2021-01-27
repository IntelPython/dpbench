#define KNEIGHBORS 5
#define DATADIM 16
#define classesNum 3


void push_queue(double* queueData, int* queueLabels, double newDistance, int newLabel, int index)
{

    while (index > 0 && newDistance < queueData[index - 1])
    {
        queueData[index] = queueData[index - 1];
        queueLabels[index] = queueLabels[index - 1];
        --index;

        queueData[index] = newDistance;
        queueLabels[index] = newLabel;
    }

}


void sort_queue(double* queueData, int* queueLabels)
{
    for (int i = 1; i < KNEIGHBORS; i++)
    {
        push_queue(queueData, queueLabels, queueData[i], queueLabels[i], i);
    }
}

double euclidean_dist(global double* x1, global double* x2)
{
    double distance = 0.0;

    for (int i = 0; i < DATADIM; ++i)
    {
        double diff = x1[i] - x2[i];
        distance += diff * diff;
    }

    double result = sqrt(distance);

    return result;
}

int simple_vote(int* queueLabels)
{
    int votesToClasses[classesNum];

    for (int i = 0; i < classesNum; ++i)
    {
        votesToClasses[i] = 0;
    }

    for (int i = 0; i < KNEIGHBORS; ++i)
    {
        votesToClasses[queueLabels[i]]++;
    }

    int maxInd = 0;
    int maxValue = 0;

    for (int i = 0; i < classesNum; ++i)
    {
        if (votesToClasses[i] > maxValue)
        {
            maxValue = votesToClasses[i];
            maxInd = i;
        }
    }

    return maxInd;
}

void kernel run_knn(global double* train, int trainSize, global int* labels, global double* test, int testSize, global int* predictions)
{
    int id = get_global_id(0);

    double queueNeighbors[KNEIGHBORS];
    int queueLabels[KNEIGHBORS];

    //count distances
    for (int j = 0; j < KNEIGHBORS; ++j)
    {

        double dist = euclidean_dist(train + j * DATADIM, test + id * DATADIM);
        queueNeighbors[j] = dist;
        queueLabels[j] = labels[j];

    }

    sort_queue(queueNeighbors, queueLabels);

    for (int j = KNEIGHBORS; j < trainSize; ++j)
    {
        double newDist = euclidean_dist(train + j * DATADIM, test + id * DATADIM);
        int newLabel = labels[j];

        double a = queueNeighbors[KNEIGHBORS - 1];

        if (newDist < a)
        {
            queueNeighbors[KNEIGHBORS - 1] = newDist;
            queueLabels[KNEIGHBORS - 1] = newLabel;

            push_queue(queueNeighbors, queueLabels, newDist, newLabel, KNEIGHBORS-1);
        }

    }
    predictions[id] = simple_vote(queueLabels);
}

