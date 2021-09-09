[Itertools by Corey Schafer](https://www.youtube.com/watch?v=Qu3dThVy6KQ)

```python
import itertools

data = ["Capital", "lovely", 'fantastic', "amazing"]

indexed_data = zip(itertools.count(), data)

# Optional arguments for count : start, step (positive or negative) 


#itertools.zip_longest(range_1, range_2)
#A kind of outer join with None provided for values of shorter range.
# Create lists of repeated values with itertools.repeat and itertools.cycle (more than one value repeated)

numbers = [0, 1, 2, 3]
itertools.product(numbers, repeat=4)
(0,0,0,0)
(0,0,0,1)
(0,0,0,2)
(0,0,0,3)
(0,0,1,0)
(0,0,1,1)
product is the same as combination_with_replacement

#itertools.chain
letters = ['a', 'b', 'c', 'd']
numbers = [0, 1, 2, 3]

for item in itertools.chain(letters,numbers):
    print(item)

combined = letters + numbers + data -- appends each iterable to the prior

#Slicing lists in Itertools

result = itertools.islice(range(10), 5) #optional arguments start, stop and step

with open('test.log', 'r') as f:
    #Read only first three lines of file
    header = itertools.islice(f, 3)
    
    for line in header:
        print(line)









```


