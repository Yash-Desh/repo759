#include <iostream>
#include "msort.h"
using namespace std;

/*
Swaps 2 elements
Takes 2 integers as input parameters
Returns void
*/
void swap(int &x, int &y)
{
    int temp = x;
    x = y;
    y = temp;
}

void bubbleSort(int *arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        // for round 1 to n-1
        bool swapped = false;

        for (int j = 0; j < n - i; j++)
        {

            // process element till n-i th index
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false)
        {
            // already sorted
            break;
        }
    }
}



/*
Takes input parameters
1. the array to be sorted
2. start index of the array
3. end index of the array
Copies the sorted array back into the input array
Returns void
*/
void merge(int *arr, int s, int e)
{

    int mid = (s + e) / 2;

    int len1 = mid - s + 1;
    int len2 = e - mid;

    int *first = new int[len1];
    int *second = new int[len2];

    // copy values
    int mainArrayIndex = s;
    for (int i = 0; i < len1; i++)
    {
        first[i] = arr[mainArrayIndex++];
    }

    mainArrayIndex = mid + 1;
    for (int i = 0; i < len2; i++)
    {
        second[i] = arr[mainArrayIndex++];
    }

    // merge 2 sorted arrays
    int index1 = 0;
    int index2 = 0;
    mainArrayIndex = s;

    
    while (index1 < len1 && index2 < len2)
    {
        if (first[index1] < second[index2])
        {
            arr[mainArrayIndex++] = first[index1++];
        }
        else
        {
            arr[mainArrayIndex++] = second[index2++];
        }
    }

    while (index1 < len1)
    {
        arr[mainArrayIndex++] = first[index1++];
    }

    while (index2 < len2)
    {
        arr[mainArrayIndex++] = second[index2++];
    }

    delete[] first;
    delete[] second;
}

void mergeSort(int *arr, int s, int e, int t)
{

    // base case
    if (s >= e)
    {
        return;
    }

    int mid = (s + e) / 2;

    // sort the left half
    if (mid - s +1 >= t)
    {
        #pragma omp task
        //cout<<"merge sort called\n";
        mergeSort(arr, s, mid, t);
    }
    else
    {
        //cout<<"Bubble sort called\n";
        bubbleSort(arr+s, mid - s + 1);
    }

    // sort the right half
    if(e-mid >= t)
    {
        #pragma omp task
        //cout<<"merge sort called\n";
        mergeSort(arr, mid + 1, e, t);
    }
    else
    {
        //cout<<"Bubble sort called\n";
        bubbleSort(arr+mid+1, e-mid);
    }

    // merge
    #pragma omp taskwait
    merge(arr, s, e);
}

void msort(int *arr, const std::size_t n, const std::size_t threshold)
{
    #pragma omp single
    {   
        //cout<<"Number of threads in msort "<<omp_get_num_threads()<<endl;
        mergeSort(arr, 0, n, threshold);
    }

    // bubbleSort(arr, n);
}

