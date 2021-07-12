
iris=load('data/iris.txt'); y=iris(:,end); X=iris(:,1:end-1);
[X, y] = shuffleData(X,y); 
[Xtr, Xva, Ytr, Yva] = splitData(X,y, .75);

for k=[1 5 10 20]
   knn = knnClassify( Xtr(:,1:2),Ytr, k);
   plotClassify2D( knn, Xtr(:,1:2), Ytr);
   fname = sprintf('hw1_4a_%d.eps',k);
   set(gca,'fontsize',20);
   print(fname,'-depsc2'); system(['epstopdf ' fname]); system(['rm ' fname]); 
end;

K=[1,2,5,10,50,100,200]; 
for k=1:length(K) 
    learner = knnClassify( Xtr(:,1:2),Ytr, K(k) ); 
    Yhat = predict( learner, Xtr(:,1:2) ); 
    etrain(k) = mean( Yhat ~= Ytr ); 
    Yhat = predict( learner, Xva(:,1:2) ); 
    evalid(k) = mean( Yhat ~= Yva ); 
end; 
figure; semilogx(K,etrain,'r-',K,evalid,'g-','linewidth',3); 
set(gca,'fontsize',20); 

Xn = X - repmat(mean(X),[148,1]);