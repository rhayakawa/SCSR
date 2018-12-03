function [x_est,arrSER] = SCSR_CL1(y,A,arrC,matQ,invMat,lambda,rho,nIteration,x_true);
%SCSR_CL1 SCSR optimization using complex L1 norm for complex discrete-valued vector 
%
% Input
%   y: measurement vector
%   A: measurement matrix
%   arrC: array for candidate of unknown variables
%   matQ: coefficients in W-SCSR optimization
%   invMat: inversion matrix in the algorithm
%   lambda, rho: parameters in the algorithm
%   nIteration: number of iterations
%   x_true: true unknown vector (only used in the evalution of SER)
%
% Output
%   x_est: estimate of unknown vector
%   arrSER: array of symbol error rate (SER)
%
    
  [M,N]=size(A);
  L=length(arrC);
  
  x_MF=A'*y;
  Phi=kron(ones(L,1),eye(N));
  thre=reshape(matQ,N*L,1)/(2*rho);
  orig=kron(arrC.',ones(N,1));
  z=zeros(L*N,1);
  r=zeros(L*N,1);
  arrSER=zeros(1,nIteration);
  for k=1:nIteration
    s=invMat*(rho*Phi.'*(z-r)+lambda*x_MF);
    u=Phi*s+r;
    prox=max(abs(u-orig)-thre,0).*(u-orig)./abs(u-orig);
    z=orig+prox;
    r=r+Phi*s-z;
    arrSER(k)=nnz(quantize(s,arrC)-x_true)/N;
  end
  x_est=s;
    
end