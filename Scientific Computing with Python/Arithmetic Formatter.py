#!/usr/bin/env python
# coding: utf-8

# In[250]:


def arithmetic_arranger(problems, result = False):
    for values in problems:
        numbers = values.split(' ')
        if len(problems) > 5:
            raise Exception('Error: Too many problems.')
        elif ('*' or '/') in values:
            raise Exception('Error: Operator must be "+" or "-".')
        elif (numbers[0].isdigit() and numbers[2].isdigit()) != True:
            raise Exception('Error: Numbers must only contain digits.')
        elif len(numbers[0]) > 4 or len(numbers[2]) > 4:
            raise Exception('Error: Numbers cannot be more than four digits.')
    
    linha1 = ''
    linha2 = ''
    
    for values in problems:
        numero = values.split(' ')
        espaco1 = ' ' * (len(numero[2]) - len(numero[0]))
        espaco2 = ' ' * (len(numero[0]) - len(numero[2]))
        linha1 = linha1 + '  ' + espaco1 + numero[0] + (' '*4)
        linha2 = linha2 + numero[1] + ' ' + espaco2 + numero[2] + (' '*4)
    
    
    arranged_problems = linha1 + '\n' + linha2
    
    return arranged_problems


# In[251]:


print(arithmetic_arranger(["22 + 2298", "3801 - 2", "45 + 666", "5555 + 43", "45 + 43"]))


# In[158]:


teste =["30 + 698", "3801 - 2", "45 + 43"]


# In[166]:


for i in teste:
    a = i.split(' ')
    print(max(a))


# In[226]:


espaco1 = (2 - 6)* ' '
espaco1


# In[232]:


print('coisa qualquer'+ '  ' + espaco1 + 'Outra coisa mcarc')


# In[ ]:




