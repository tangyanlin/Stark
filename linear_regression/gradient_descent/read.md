#Gradient Descent

* batch gradient descent

  > repeat until convergence:

  >> <img src="http://chart.googleapis.com/chart?cht=tx&chl= \normalsize \Theta_i := \Theta_i - \alpha \displaystyle\sum_{j=1}^m (h_\Theta(x^{(j)})-y^{(j)}) \cdot x_i^{(j)}" style="border:none;">
  
* stochastic gradient descent

 > Repeat {

 >>For j=1 to m {
  
 >>><img src="http://chart.googleapis.com/chart?cht=tx&chl= \normalsize \Theta_i := \Theta_i - \alpha  (h_\Theta(x^{(j)})-y^{(j)}) \cdot x_i^{(j)}" style="border:none;">
  
 >>}
  
 >}
