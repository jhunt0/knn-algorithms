# Author: Jacob B. Hunt
# Project for MATH 4/6686: Statistical Learning II

# knn()

# Parameters:
# k.vec: vector of k-values to try with KNN
# x.train: predictor matrix
# y.train: response vector
# x.test: predictor matrix

# Returns: 
# Matrix of predictions with |k.vec| columns and |rows(x.test)| rows

knn = function(k.vec, x.train, y.train, x.test){
  
  x.train = as.matrix(x.train)
  x.test = as.matrix(x.test)
  y.train = as.vector(t(y.train))
  
  
  results = matrix(0, nrow(x.test), length(k.vec))
  errors = matrix(0, nrow(x.test), length(k.vec))
  distances = matrix(0, nrow(x.train), 3)
  
  for (i in 1:nrow(x.test)){
    for (j in 1:nrow(x.train)){
      
      # get distance between test point i and train point j
      distances[j,] = c( dist(rbind(c(x.test[i,]), c(x.train[j,]))), y.train[j], j)
    }
    
    ordered = distances[order(distances[,1]),]
    
    for (k in 1:length(k.vec)){
      results[i,k] = mean(ordered[1:k.vec[k],2])
    }
  }
  return(results)
}


# knn.cv()

# Parameters:
# k.vec: vector of k-values to try with KNN
# x.train: predictor matrix
# y.train: response vector

# Returns:
# Matrix of predictions with |k.vec| columns and |rows(x.train)| rows

knn.cv = function(k.vec, x.train, y.train){
  
  x.train = as.matrix(x.train)
  y.train = as.vector(t(y.train))
  
  results = matrix(0, nrow(x.train), length(k.vec))
  
  for (i in 1:nrow(x.train)){
    results[i,] = as.vector( t( c(knn(k.vec, x.train[-i,], c(y.train[-i]), x.train[i] )) ) )
  }
  return(results)
}

# knn.sel()

# Parameters:
# k.vec: vector of k-values to try with KNN
# x.train: predictor matrix 
# y.train: response vector 
# x.test: predictor matrix

# Reutrns:
# Outputs of a linear model that uses as it's response variables the
# predictions from KNN for k-values suppllied in k.vec

knn.sel = function(k.vec, x.train, y.train, x.test){
  
  x.train = as.matrix(x.train)
  x.test = as.matrix(x.test)
  y.train = as.vector(t(y.train))
  
  results = matrix(0, nrow(x.test), 1)
  distances = matrix(0, nrow(x.train), 3)
  
  for (i in 1:nrow(x.test)){    
    for (j in 1:nrow(x.train)){   
      distances[j,] = c( dist(rbind(c(x.test[i,]), as.vector(x.train[j,]))), y.train[j], j)
    }
    
    ordered = distances[order(distances[,1]),]
    
    min = Inf
    best.k = 0
    
    for (k in k.vec){
      
      # pass the k closest neighbors of j to cv.knn to obtain a vector of predictions
      
      index = ordered[1:k,3]
      
      cv.x.train = as.matrix((x.train[index,]))
      cv.y.train = y.train[index]
      
      predictions = knn.cv(c(k-1), cv.x.train, cv.y.train)
      error = mean(predictions - cv.y.train)^2
      
      if (error < min || is.na(error)==TRUE)
        if (is.na(error))
          min=0
      else
        min = error
      best.k = k
    }
    
    best.index = ordered[1:best.k,3]
    
    knn.x.train = as.matrix(x.train[best.index,])
    knn.y.train = y.train[best.index]
    
    results[i,] = knn(best.k, knn.x.train, knn.y.train, t(as.matrix(x.test[i,])))
    
  }
  return(results)
}

# knn.pool()

# Parameters:
# k.vec: vector of k-values to try with KNN
# x.train: predictor matrix 
# y.train: response vector 
# x.test: predictor matrix

# Returns: Vector of predictions consisting of trained, linear combinations of the results
# of knn done with the supplied k-values.

# a vector of predictions for the test cases using the combination of the predictions from LOOCV via linear regression

knn.pool = function(k.vec, x.train, y.train, x.test){
  x.train = as.matrix(x.train)
  x.test = as.matrix(x.test)
  y.train = as.vector(t(y.train))
  # First we get a series of predictions for each train point
  predictions = knn.cv(k.vec, x.train, y.train)
  # and the test points
  test.predictions = knn(k.vec, x.train, y.train, x.test)
  # then train a Linear Regression model 
  knn.lm = lm(y.train~., data=as.data.frame(predictions))
  print(summary(knn.lm))
  return(predict(knn.lm, as.data.frame(test.predictions)))
}