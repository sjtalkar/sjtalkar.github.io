# Understanding Power BI Key Influencers

> - Reference on Github: [BlueGranite/AI-in-a-Day](https://github.com/BlueGranite/AI-in-a-Day/tree/master/AI%20visuals%20-%20Key%20Influencers)
> - [Video](https://www.bluegranite.com/blog/exploring-power-bis-key-influencers) 

1. The target that is supported currently in Power BI has to be categorical (Example: What influences Rating to be High/Low)
2. In the example provided, you can select the potential influencers which can be Quantitative (size of company) or Categorical (theme - login screen, features/navigation/reliabiity), Nominal - customer/administrator/publisher  or Ordinal (Basic/Premier/Deluxe)
3. For each selected feature a graph is shown where a specific value within that feature influenced the target the most. The average influence of the other values is shown as a constant line and so how much more the most influential value affected the target can be calculated by dividing the percentage influence by average influence of the rest of the values.
4. For featurs that are quantitatve rather than categorical, the visual might also choose to bin the quantitive values into ranges such as 0 or less, 0-29, 30 or morem to get insights such as to how longer tenure (time of association with company) affects the Ranking.
5. For other quantitative variable, Power BI might elect to show the relationship as a regression line.
6. The influencers can also be measures such as Count of support tickets per customer. In this particular scenario all the granularity is at customer level.
7. The visual also has a Top Segments feature which creates "clusters" called segments with percentage influence on the target. The size of the cluster/segment is scaled by the number of data points in the cluster.
8. Clicking on the cluster/segment gives you information about how many percentage points above the average the segment is in selecting the target category. It also provides the count of data points in that cluster.
9. 



