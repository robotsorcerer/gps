I love your modernization analysis. Good work! I think in addtiion to using     
     move sematics as you migrate to C++1z, you should also consider some aspects of                      
     C++20. In your testing, since ros 1.x is not available on the host device, you                       
     should consider pulling from a docker image, build all the necessary                                 
     dependencies and rigorously test the development of your migration. Be sure to                       
     make your migration strategy iterative and validate every changes                                    
     incrementally!  