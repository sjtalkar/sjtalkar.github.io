# How to create a function to piece page results of an API call together

[Gil Raviv on APIs and advanced features of Power BI](https://www.youtube.com/watch?v=r0Qk5V8dvgg)


### API URL In Power BI: Get Data from Web and paste URL
`https://swapi.dev/api/?page=1`


### How to create a list in Power BI Blank Query
```mquery
= {1..20}
```
Right click on above list and convert To Table

> Did you know right clicking on columns and values and selecting the below command,creates a query with that column name and those column values?
`Add as New Query`

In our application, the Source query step will have a count field with the number of pages for people. Right click on it and create a query out of it.

Using the below a dynamic range list can be created, where PeopleCount can be a parameter or a Query.

`= List.Numbers(1,PeopleCount)`

### Generalizing a repeated set of steps

- Create a parameter : NumberOfPage initialized to 1
- Use it in the source, the API call 
`= Json.Document(Web.Contents("https://swapi.dev/api/people/?page=" & NumberOfPage))`
- Right click on the query you want to convert to a function and select **Create Function** 
- Invoke the Function by 
    - Access the table containing a column full of values of the same type as the parameter to the function.
    - In the top menu select Add Column
    - Select Invoke Custom Function (next to Create Column from example, Conditional ....)
    - Select the function created earlier
    - Select the column in the table that will serve as Parameter
    - If you run into Privcy issue, In Options and Settings look into Privacy (for current file)
    https://community.powerbi.com/t5/Desktop/Formula-Firewall-Query-references-other-queries-so-it-may-not/td-p/18619
    - The SingleQuery (the sample query with one page can continue to support modifications and it will flow to all the tables that the function creates)

### Using a What If parameter

[Modeling --> New Parameter](https://docs.microsoft.com/en-us/power-bi/transform-model/desktop-what-if)




