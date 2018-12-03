function y = quantize(x,arrC)
  N=length(x);
  L=length(arrC);
  matDist=abs(x*ones(1,L)-ones(N,1)*arrC);
  [~,indeces]=min(matDist.');
  y=arrC(indeces).';
end
