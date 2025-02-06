import numpy as np
from numpy.random import default_rng
from random import randint

print(np.ones((2,5),dtype = int))

print(np.zeros((2,5),dtype= float))
print(np.arange(2,5,1))
print(np.linspace(2,10,3))
print(np.arange(1, 9, 2).reshape(2,2))
print()
print(np.arange(2,10,2).reshape(2,2))
print(np.ones(5,int))
print(np.zeros(100,int).reshape(10,10))
print(np.identity(10).reshape(5,20).T)

print(np.linspace(1,100, 10))

#random number generators:
print("----------- random number generators--------------\n")

rngg = default_rng(47)
rng = np.random.default_rng(616)
print(rngg.random(10))
print(rngg.normal(10,2,100))
print()
array = [randint(1,100) for i in range(1,101)]

print(np.array(array).reshape(10,10))

#--------------- Indexing & Slicing Arrays ----------------------
print()
import numpy as np

integer_array = np.arange(12)

print(integer_array)
print(integer_array[::5])

new_array = integer_array.reshape(3,4)
print(new_array[:,:])
print()
print(new_array[:,1:])
print(new_array[2,1])


ages = np.array([5, 10, 15, 19, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
youth_ages = ages[1:4]
print(youth_ages)
adult_ages = ages[5:14]
print(adult_ages)
senior_ages = ages[15:]
print(senior_ages)

rng = np.random.default_rng(2022)
random_array = rng.random(9).reshape(3, 3)
print(random_array)
print()
print(random_array[:2,:])
print()
print(random_array[:,0])
print()
print(random_array[2,1])

#------------- array operations --------------------------------------------
import numpy as np

rng = np.random.default_rng(616)

inventory = rng.integers(0,100,10)
print(inventory)
print(inventory - 24 )

print((inventory / 2).dtype)

price = ((rng.random(10))*10).round(2)
print(price)

print(price * inventory)

print(sum(price * inventory))

inventory_list = inventory.tolist()
print()
print(inventory)
print(inventory_list)
print()
print(inventory + 2)
#print(inventory_list +2)

new_inventory = []

for x in inventory_list:
    new_inventory.append(x+2)
    print(new_inventory)
print()
print(new_inventory)
print()
print([x + 2 for x in inventory_list])
print()
print("Invenotry_list:", inventory_list)
print("price:",price)
print([x * y for x, y in zip(inventory_list,price.tolist())])
#------------- assignment --------------------

import numpy as np

# Population age data for two small towns
town_a_ages = np.array([25, 45, 70, 34, 58])
town_b_ages = np.array([30, 55, 65, 40, 60])

diff_age = town_a_ages - town_b_ages
print(diff_age)

diff_age =[]

for x, y in zip(town_a_ages.tolist(),town_b_ages.tolist()):
    diff_age.append(x-y)
print(diff_age)

prices = np.array([5.99, 6.99, 22.49, 99.99,4.99,  49.99])
total = prices + 5
print(total)

#----------- filtering arrays --------------------
print()
import numpy as np

my_array = np.arange(20)
print(my_array)
print(my_array[my_array % 2 == 0])
print()
mask = my_array % 2
print(my_array[mask == 0])
even_odd = np.array(['even','odd'] * 10)

print(even_odd)

print(even_odd != 'odd')
print(even_odd[even_odd != 'odd'])

mask = even_odd != 'odd'
print()
print(even_odd[mask == True])
my_array[my_array % 2 == 0] = 0
print(my_array)

my_array[even_odd != 'odd'] = -1
print(my_array)

mask = (even_odd != 'odd') | (even_odd != 'even')
print(even_odd[mask])

#--------------Numpy Where Fuction -------------------------------

inventory_array = np.array([12, 0, 18, 0, 4])

product_array = np.array(['fruits', 'vegtables','ceral','dairy','eggs'])

inventory_status = np.where(inventory_array <=0, 'Out of Stock', 'In Stock')
inventory_status_2 = np.where(inventory_array <=0, 'Out of Stock', product_array)

print(inventory_status)
print(inventory_status_2)
#--------------- exercise ----------------------

import numpy as np
# Provided array of ages
ages = np.array([25, 12, 15, 64, 35, 80, 45, 10, 22, 55])
ages_index = []
for i in range(10):
    if ages[i] >= 18: ages_index.append(i)
print(ages_index)

adult_ages = np.where(ages >= 18)
print()
print(np.array(adult_ages)[0])


product = np.array(product_array)
product[0] = 'cola'
prices = np.array([3, 40, 8, 9, 30])

print(product)
print(prices)
product_25 = product[prices > 25]
print(product_25)

mask = (prices > 25) | (product == 'cola')

fency_feast_special = product[mask]

shipping_cost = np.where(prices > 20, 5, 0)
print(shipping_cost)

# ------------- array aggregation --------------------

import numpy as np

rng = np.random.default_rng(616)

price = (rng.random(10) * 10).round(2)

print(price)

inventory = rng.integers(0,100,10)

print(inventory)

print(price.mean())
print(inventory.sum())

print((price * inventory))
print((price * inventory).argmin())

price_2d = price.reshape(5,2)
print(price_2d)
print()
print(price_2d.sum(axis=0))
print(price_2d.sum(axis = 1))

#------------------ 
print()
import numpy as np
rng = np.random.default_rng(616)

price = (rng.random(10) * 10).round(2)
inventory = inventory = rng.integers(0,100,10)
print("price:",price)
print("inventory:", inventory)
print("sorted price: ",np.sort(price))

product_value = price * inventory
print()
print(product_value)
product_value = np.sort(product_value)
print(product_value)

reshaped_product_value = product_value.reshape(2,5)


print(np.median(reshaped_product_value))
print()
print(reshaped_product_value)
print(np.percentile(reshaped_product_value, 10))

print(np.sqrt(reshaped_product_value))
print(np.unique(reshaped_product_value))

print(np.unique(np.ones(100)))

#---------------------------------- sort practice --------------------
print()
sales_array = np.array([0,   5, 155,   0, 518, 0,1827, 616, 317, 325])
sales_array = sales_array.reshape(2,5)
print(sales_array)
print()
print(sales_array.sort())
print(product_value[0])

#------------------------ sort --------------------------------
import numpy as np

# Provided array of ages
ages = np.array([25, 12, 15, 64, 35, 80, 45, 10, 22, 55])
print(ages)
ages.sort()
print(ages)
print("min: ", ages[0])
print("max: ",ages[-1])

#---------------- Vectorization & Broadcasting ----------------------------
np.list