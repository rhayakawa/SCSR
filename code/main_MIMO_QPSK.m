%%-----------------------------------------------------------
% simulation for IW-SCSR in MIMO signal detection with QPSK
%%-----------------------------------------------------------

addpath('subfunctions');

N=50; % number of transmit antennas
M=40; % number of receive antennas
nIteration=100; % number of iterations
nSymbolVector=10; % number of unknown vectors per channel realization and SNR
nSample=5; % number of samples of channel matrix
nUpdate=5; % number of weight update
arrSNR=0:2.5:30; % array for signal-to-noise ratio

% probability distribution
L=4;
arrP=[1/4 1/4 1/4 1/4];
arrC=[1+1j -1+1j -1-1j 1-1j];
matQ_init=ones(N,L)/L;

% parameter of the proposed algorithm
beta_RL1=15;
rho=0.1;

% parameter for correlated channels
c=299792458; % [m/s]
frequency=5.0*10^(9); % [Hz]
wavelength=c/frequency; % [m]
r=0.5;
d_t=r*wavelength;
d_r=r*wavelength;

rng('shuffle');

matSumSER_SCSR_RL1=zeros(nUpdate,length(arrSNR));
for i=1:nSample
  disp(['i=' num2str(i)]);
  % channel matrix
  % [A,~]=makeChannel(M,N); % i.i.d.
  [A,~]=makeChannel_corr(M,N,d_t,d_r,wavelength); % correlated
  
  for SNRIndex=1:length(arrSNR)
    SNR=arrSNR(SNRIndex);
    disp(['  SNR=' num2str(SNR)]);
    % variance of additive noise
    sigma_c=sqrt(2*N/(10^(SNR/10)));
    % parameter of optimization problem
    numer_RL1=0;
    for l=1:L
      numer_RL1=numer_RL1+arrP(l)*sum(sum(matQ_init.*(abs(real(arrC(l)-ones(N,1)*arrC))+abs(imag(arrC(l)-ones(N,1)*arrC)))));
    end
    lambda_RL1_init=numer_RL1/(beta_RL1*M*sigma_c^(2));
    % inverse matrix
    invMat_RL1_init=(rho*L*eye(N)+lambda_RL1_init*(A'*A))^(-1);

    for symbolVectorIndex=1:nSymbolVector
      % transmitted signal vector
      x=(randi([0,1],N,1)*2-ones(N,1))+1j*(randi([0,1],N,1)*2-ones(N,1));
      % additive noise vector
      v=(randn(M,1)+1j*randn(M,1))/sqrt(2)*sigma_c;
      % received signal vector
      y=A*x+v;

      %% IW-SCSR
      matQ=matQ_init;
      invMat_RL1=invMat_RL1_init;
      lambda_RL1=lambda_RL1_init;
      for itrIndex=1:nUpdate
        [x_est,~]=SCSR_RL1(y,A,arrC,matQ,invMat_RL1,lambda_RL1,rho,nIteration,x);
        matSumSER_SCSR_RL1(itrIndex,SNRIndex)=matSumSER_SCSR_RL1(itrIndex,SNRIndex)+nnz(quantize(x_est,arrC)-x)/N;
        matD=abs(x_est*ones(1,L)-ones(N,1)*arrC);
        matQ=matD.^(-1)./(sum((matD.^(-1)),2)*ones(1,L));
        % parameter of optimization problem
        numer_RL1=0;
        for l=1:L
          numer_RL1=numer_RL1+arrP(l)*sum(sum(matQ.*(abs(real(arrC(l)-ones(N,1)*arrC))+abs(imag(arrC(l)-ones(N,1)*arrC)))));
        end
        lambda_RL1=numer_RL1/(beta_RL1*M*sigma_c^(2));
        invMat_RL1=(rho*L*eye(N)+lambda_RL1*(A'*A))^(-1);
      end
    end
  end
end
matSER_SCSR_RL1=matSumSER_SCSR_RL1/nSample/nSymbolVector;

%% Display results
arrMarker=['o';'^';'s';'d';'v';'*';'<';'x';'d';'p';'h';];
close all;
figure;
setLegend={};
for itrIndex=1:4:nUpdate
  h=semilogy(arrSNR,matSER_SCSR_RL1(itrIndex,:),['-' arrMarker(itrIndex)],'LineWidth',1,'MarkerSize',8);
  setLegend=[setLegend ['IW-SCSR ($T=' num2str(itrIndex) '$)']];
  hold on;
end
grid on;
objLegend=legend(setLegend,'Location','northeast');
objLegend.Interpreter='latex';
objLegend.FontSize=16;
fig=gca;
fig.FontSize=18;
fig.TickLabelInterpreter='latex';
fig.XLabel.Interpreter='latex';
fig.YLabel.Interpreter='latex';
xlabel('SNR (dB)');
ylabel('SER');
axis([arrSNR(1) arrSNR(length(arrSNR)) 1e-5 1]);
saveas(h, 'MIMO_QPSK.eps', 'epsc');
