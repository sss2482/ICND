from scipy.stats import beta
from scipy.integrate import quad

class ProbDistribution:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def value(self, x):
        return beta.pdf(x, self.alpha, self.beta)
    
    def area_under_pdf(self, x, y):
        """
        Calculate the area under the Beta distribution's PDF in the range [x - y, x + y].
        
        Parameters:
        - alpha: Shape parameter alpha of the Beta distribution
        - beta_param: Shape parameter beta of the Beta distribution
        - x: The center value for the range
        - y: The half-width of the range
        
        Returns:
        - Area under the Beta distribution's PDF in the range [x - y, x + y]
        """
        
        # Calculate the integral (area) in the range [x - y, x + y]
        rl = min([1, x+y])
        ll = max([0, x-y])
        area, _ = quad(self.value, ll, rl)
        
        return area
    
    def prob_val(self, x):
        return self.area_under_pdf(x, 0.1)
    

pd = ProbDistribution(0.5,5)
print(pd.prob_val(0))



