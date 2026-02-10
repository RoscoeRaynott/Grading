clear all;
% close all;

% Initialize parallel pool for faster processing
if isempty(gcp('nocreate')), parpool; end

%Data creation
gap=0.02;
t=0:gap:1;
ScoSh=1;
FITTER_TYPE='poly';%'pava';%

% --- Setup Sample Sizes to Test ---
n_vals = 40:20:620; % 30 equally spaced values, all divisible by 4
n_sample_sizes = length(n_vals);
n_cv_folds = 2; % Number of cross-validation folds (change cross_v loop to 1:4 for all folds)
results_table = []; % Will store [n, cross_v, trainR2, testR2]
row_idx = 0;
n_repeats = 5; % Number of independent dataset repeats per sample size

% Huber loss scale factor (delta = huber_scale * stdER)
huber_scale = 1.5;  % Tune this multiplier

is_g=1;
is_FOU=1;
is_Clustering=0;
Class1=1;
Class2=0;%Class1/2;
v=[sqrt(2) 1 sqrt(2) 1 sqrt(2) 1]*sqrt(2);
        
beta=@(t) v(1)*sin(2*pi*t)+v(2)*cos(2*pi*t)+v(3)*sin(4*pi*t)+v(4)*cos(4*pi*t)+v(5)*sin(6*pi*t)+v(6)*cos(6*pi*t);
h=@(t) t;%t.^3+5*t;%3*t+4;%-3*t.^3;%t.^5+6*t.^3;%tan(t);%log(t);%t.^5+-t;%
g=@(t) t.^2;%t.^3-3.*t;%0;%t;%5*t.^2-4;%10.*t-4;%

const=1;

%hcentering
hCENTERING=0;

if hCENTERING==1
    x=x/norm(x);
end

%gcentering
gCENTERING=0;

%% Loop over Sample Sizes
fprintf('Starting comparison over %d sample sizes...\n', n_sample_sizes);

for s_idx = 1:n_sample_sizes
    % --- Stop flag: create a file called STOP.flag to break cleanly ---
    if exist('STOP.flag', 'file'), fprintf('STOP flag detected. Breaking.\n'); delete('STOP.flag'); break; end

    n = n_vals(s_idx);
    stdER = 0.5;
    fprintf('\nProcessing n = %d ...\n', n);

    for rep = 1:n_repeats

    % Adaptive Huber loss: delta scales with noise level
    huber_delta = max(0.1, huber_scale * stdER);
    huber_loss = @(e) sum((abs(e) <= huber_delta) .* (0.5 * e.^2) + ...
                          (abs(e) > huber_delta) .* (huber_delta * (abs(e) - 0.5*huber_delta)));

    % Clear variables that change size with n
    clear yte_pred ytest_store epsi inp;

    % --- Data generation for this sample size ---
    y=zeros(n,1);
    gamSync=zeros(n,size(t,2));
    costtrue=0;
    f=zeros(n,length(t));
    qi=zeros(n,length(t));
    fi0 = cell(1,n); gammai = cell(1,n); fi = cell(1,n);

    % Generate curves and initial q representations
    for i=1:n
        ci1=normrnd(0,1);
        ci2=normrnd(0,1);
        ci3=normrnd(0,1);
        ci4=normrnd(0,1);
        fi0{i}=@(t) ci1*sqrt(2)*sin(2*pi*t)+ci2*sqrt(2)*cos(2*pi*t);
        alpha=unifrnd(-1,1);
        gammai{i}=@(t) t+alpha.*(t).*(t-1);
        fi{i}=@(x) fi0{i}(gammai{i}(x));
        f(i,:)=fi{i}(t);
        qi(i,:)=curve_to_q(fi{i}(t));
    end

    % Parallel registration with beta if ScoSh==1
    if ScoSh==1
        beta_t = beta(t);
        parfor i=1:n
            gam_i = DynamicProgrammingQ_Adam(qi(i,:), beta_t, 0, 0);
            gamSync(i,:) = gam_i;
            dumf = interp1(t, f(i,:), gam_i, 'spline');
            qi(i,:) = curve_to_q(dumf);
        end
    end

    % Compute x (inner products with beta)
    beta_t = beta(t);
    x = zeros(1,n);
    for i=1:n
        x(i)=gap*dot(beta_t,qi(i,:));
    end
    ht=min(x):0.1:max(x);
    gt=min(f(:,1)):0.1:max(f(:,1));

    % --- Generate y with fixed noise ---
    epsi = zeros(1,n);

    for i=1:n
        epsi(i)=normrnd(0,stdER);
%     %Simulation y_i
%     y(i)=epsi(i)+h(x(i));
%     if is_g==1
%         y(i)=y(i)+g(f(i,1));
%     end
    
    %Semi Real y_i
        y(i)=epsi(i)+max(f(i,:))-min(f(i,:));%sum(abs(diff([0 fi{i}(t)])/gap)*gap);%
        costtrue=costtrue+epsi(i)^2;
    end
if is_Clustering==1
    my=mean(y);
    for i=1:n
        if y(i)>my
            y(i)=Class1;
        else
            y(i)=0;
        end
    end
end

%% Clustering Daatset
if is_Clustering==2
    for i=1:n/2
        ci1=normrnd(0,1);
        ci2=normrnd(0,1);
        fi0{i}=@(t) abs(ci1)*sqrt(2)*sin(1*pi*t);%+ci2*sqrt(2)*cos(2*pi*t);
        alpha=unifrnd(-1,1);
        gammai{i}=@(t) t+alpha.*(t).*(t-1);
        fi{i}=@(x) fi0{i}(gammai{i}(x));
        f(i,:)=fi{i}(t)/norm(fi{i}(t));
        qi(i,:)=curve_to_q(fi{i}(t));
        y(i)=Class2;
        ci1=normrnd(0,1);
        ci2=normrnd(0,1);
        ci3=normrnd(0,1);
        ci4=normrnd(0,1);
        fi0{n/2+i}=@(t) ci1*sqrt(2)*sin(3*pi*t);%+ci4*sqrt(2)*cos(4*pi*t);%ci2*sqrt(2)*cos(2*pi*t);%+ci3*sqrt(2)*sin(4*pi*t)+
        alpha=unifrnd(-1,1);
        gammai{n/2+i}=@(t) t+alpha.*(t).*(t-1);
        fi{n/2+i}=@(x) fi0{n/2+i}(gammai{n/2+i}(x));
        f(n/2+i,:)=fi{n/2+i}(t)/norm(fi{n/2+i}(t));
        qi(n/2+i,:)=curve_to_q(fi{n/2+i}(t));
        y(n/2+i)=Class1;
    end
    mee=sum(qi(1:5,:)+qi((n/2+1):(n/2+5),:))/10;
    for i=1:10
        figure(901);
        plot(t,f(i,:),'LineWidth',2,'Color','g');
        hold on;
        plot(t,f(n/2+i,:),'LineWidth',2,'Color','b');
        gaaaama=DynamicProgrammingQ_Adam(qi(i,:),mee,0,0);
        gaaaaama=DynamicProgrammingQ_Adam(qi(n/2+i,:),mee,0,0);
        figure(903);
        plot(t,interp1(t,f(i,:),gaaaama,'spline'),'g','LineWidth',2);
        hold on;
        plot(t,interp1(t,f(n/2+i,:),gaaaaama,'spline'),'b','LineWidth',2);
        figure(902);
        plot(t,gaaaama,'g','LineWidth',2);
        hold on;
        plot(t,gaaaaama,'b','LineWidth',2);
    end
    figure(903);
    plot(t,mee,'r');
    figure(904)
    for i=1:5
        figure(904);
        plot(i,0,'b.');
        hold on;
        plot(5+i,1,'g.');
        
    end

    figure(904);
    plot(10,1.3,'.',2,-0.2,'.');
    save_pixels('BLURR.png',602,470);
end
    
% % %% SpanishWeatherDAta Experiment
% %       
% % for Data1=1
% % load("Data1.mat");
% % index=randperm(size(X,1),size(X,1));
% % %X=X(:,1:3:365);
% % s=zeros(1,31);
% % Xo=X(:,1:5:365);
% % for J=30
% %     X=Xo;
% %     gap=1/(size(X,2)-1);
% %     t=0:gap:1;
% %     %J=21;
% %     % basis functions
% %     for i=1:(J-1/2)
% %         basis{1}=@(t) ones(1,size(t,2))/norm(ones(1,size(t,2)));
% %         for j=1:i
% %             basis{2*j}=@(t) sin(2*pi*j*t)/norm(sin(2*pi*j*t));
% %             basis{2*j+1}=@(t) cos(2*pi*j*t)/norm(cos(2*pi*j*t));
% %         end
% %     end
% %     
% %     
% %     for k=1:size(X,1)
% %         plott=zeros(size(t));
% %         for i=1:J
% %             plott=plott+X(k,:)*basis{i}(t)'*basis{i}(t);%/(norm(basis{i}(t))*norm(beta(t)));
% %         end
% %         
% % %         figure(91);
% % %         plot(t,X(k,:),'+-',t,plott,'.');
% %         %s(J)=s(J)+norm(X(k,:)-plott);
% %         X(k,:)=plott;
% %     end
% % %     s(J)
% % end
% % figure(91);
% % plot(X','.-');
% % end
% % f=X;
% % y=Y;
% % n=size(X,1);

% %  %% hvd
% %        
% % dataX=readtable('current-covid-patients-hospital.csv');
% % Countries=string(unique(dataX.Entity));
% % indexx=1;
% % CouSet=string(dataX.Entity);
% % LT=NaT(1,size(Countries,1));
% % HT=NaT(1,size(Countries,1));
% % while indexx<=size(Countries,1)
% %     arr=find(CouSet==Countries(indexx));
% %     LT(indexx)=dataX.Day(arr(1),:);
% %     HT(indexx)=dataX.Day(arr(end),:);
% %     indexx=indexx+1;
% % end
% % 
% % gap=1/daysact(min(LT),max(HT));
% % t=0:gap:1;
% % X=zeros(size(Countries,1),length(t));
% % for i=1:size(Countries,1)
% %     zeross=daysact(min(LT),LT(i));
% %     arr=find(CouSet==Countries(i));
% %     X(i,(zeross+1):(zeross+length(arr)))=dataX.DailyHospitalOccupancy(arr);
% %     ff=find(X(i,:)>0);
% %     kl=X(i,ff(1));
% %     X(i,1:(ff(1)-1))=(kl/(ff(1)-1)):(kl/(ff(1)-1)):kl;
% % %     X(i,:)=X(i,:)/norm(X(i,:));
% % end
% % X=X(:,1:10:1019);
% % 
% % dataY=readtable('owid-covid-dataLatest.xlsx');
% % Y=zeros(size(Countries,1),1);
% % CouSetY=string(dataY.location);
% % for i=1:size(Countries,1)
% %     arr=find(CouSetY==Countries(i));
% %     Y(i)=dataY.total_deaths(arr(end));
% % end
% % 
% % X(isnan(Y),:)=[];
% % Countries(isnan(Y))=[];
% % Y(isnan(Y))=[];
% % 
% % X(end,:)=[];
% % Y(end)=[];
% % Countries(end)=[];
% % const=1;
% % Y=Y/const;


% % %% NCVH
% % 
% % dataX=readtable('current-covid-patients-hospital.csv');
% % Countries=string(unique(dataX.Entity));
% % indexx=1;
% % CouSet=string(dataX.Entity);
% % LT=NaT(1,size(Countries,1));
% % HT=NaT(1,size(Countries,1));
% % while indexx<=size(Countries,1)
% %     arr=find(CouSet==Countries(indexx));
% %     LT(indexx)=dataX.Day(arr(1),:);
% %     HT(indexx)=dataX.Day(arr(end),:);
% %     indexx=indexx+1;
% % end
% % 
% % gap=1/daysact(min(LT),max(HT));
% % t=0:gap:1;
% % X=zeros(size(Countries,1),length(t));
% % for i=1:size(Countries,1)
% %     zeross=daysact(min(LT),LT(i));
% %     arr=find(CouSet==Countries(i));
% %     X(i,(zeross+1):(zeross+length(arr)))=dataX.DailyHospitalOccupancy(arr);
% %     ff=find(X(i,:)>0);
% %     kl=X(i,ff(1));
% %     X(i,1:(ff(1)-1))=(kl/(ff(1)-1)):(kl/(ff(1)-1)):kl;
% % end
% % Y=sum(X');
% % CouSetY=Countries;
% % 
% % dataX=readtable('owid-covid-dataLatest.xlsx');
% % CouSetX=string(unique(dataX.location));
% % indexx=1;
% % LT=NaT(1,size(CouSetX,1));
% % HT=NaT(1,size(CouSetX,1));
% % while indexx<=size(CouSetX,1)
% %     arr=find(dataX.location==CouSetX(indexx));
% %     LT(indexx)=dataX.date(arr(1),:);
% %     HT(indexx)=dataX.date(arr(end),:);
% %     indexx=indexx+1;
% % end
% % 
% % gap=1/daysact(min(LT),max(HT));
% % t=0:gap:1;
% % X=zeros(size(CouSetX,1),length(t));
% % notworthit=0;
% % for i=1:size(CouSetX,1)
% %     zeross=daysact(min(LT),LT(i));
% %     arr=find(dataX.location==CouSetX(i));
% %     X(i,(zeross+1):(zeross+length(arr)))=dataX.new_cases(arr);
% %     ff=find(X(i,:)>0);
% %     if (~isempty(ff))
% %         kl=X(i,ff(1));
% %         X(i,1:(ff(1)-1))=(kl/(ff(1)-1)):(kl/(ff(1)-1)):kl;
% %     else
% %         notworthit=[notworthit i];
% %     end
% % end
% % notworthit(1)=[];
% % X(notworthit,:)=[];
% % CouSetX(notworthit)=[];
% % X(isnan(X))=0;
% % 
% % [CouSetXY,ndx]=intersect(CouSetX,CouSetY,'stable');
% % X=X(ndx,:);
% % [CouSetXY,ndxY]=intersect(CouSetY,CouSetXY,'stable');
% % Y=Y(ndxY)';
% % 
% % for i=1:size(X,1)
% %     filter=ones(1,10)/10;
% %     X(i,:)=conv(X(i,:),filter,'same');
% % end
% % % for i=1:size(X,1)
% % % %     Y(i)=Y(i)/norm(X(i,:));
% % %     X(i,:)=X(i,:)/norm(X(i,:)); 
% % % end
% % X=X(:,1:10:1024);
% % % plot(X');
% % % pause();
% % % 
% % const=1000000;
% % Y=Y/const;

% f=X;
% y=Y;
% n=size(X,1);
% gap=1/(size(X,2)-1);
% t=0:gap:1;

% h_calc(x,y,6,76,FITTER_TYPE);

%% ConfidenceInterval

n_ci=1;
betamat=zeros(n_ci,length(t));
hmat=zeros(n_ci,length(ht));
gmat=zeros(n_ci,length(gt));
for ci=1:n_ci
    
    %Training & cross validation
    ind=randperm(n,n);
    testR2=zeros(1,4);
    trainR2=zeros(1,4);

    for cross_v=1:n_cv_folds
    %     clear yihat yihat2 qihat qihat2 gam inp ytr_pred yte_pred
        cross_v
        %TestTrain Breakup
        testind=ind(((cross_v-1)*n/4+1):(cross_v*n/4));
        trainind=setdiff(ind,testind);
        ytest=y(testind);
        ytrain=y(trainind);
        ftest=f(testind,:);
        ftrain=f(trainind,:);
        ntrain=length(trainind);
        costtrain=sum(epsi(trainind).^2);

        %Minimization 
        Cost_mini_style=2;
        beta_rep=10;
        hhat=@(x) x(:);%(x(:)).^3;%
        ghat=@(x) 0;
        J=5;
        c=0.1*ones(1,J);
        REG=1;
        No_h=0; %1 for No index, 0 for index
        if No_h==0
            h_rep=5;
        else
            h_rep=1;
            hhat=@(x) x(:);
        end
        pp=2;
        pp_g=2;

        %basis
        if is_FOU==1
            for i=1:(J-1/2)
                basis{1}=@(t) ones(1,size(t,2))/norm(ones(1,size(t,2)));
                for j=1:i
                    basis{2*j}=@(t) sin(2*pi*j*t)/norm(sin(2*pi*j*t));
                    basis{2*j+1}=@(t) cos(2*pi*j*t)/norm(cos(2*pi*j*t));
                end
            end
        else
            belements=FPCA(f,J);
            for i=1:J
                basis{i}=@(x) interp1(t,belements(:,i),x);
            end
        end
        % iter_cnt=1;
        val=zeros(1,beta_rep*h_rep);
        if Cost_mini_style==2
            for i=1:h_rep
                for j=1:beta_rep
                    %beta init
                    betahat=@(t) c(1)*basis{1}(t);
                    for k=2:J
                        betahat=@(t) betahat(t)+ c(k)*basis{k}(t);
                    end
    %                 if i>1
    %                     hhat=@(x) (x'.^(0:pp)*p')'; 
    %                 end
                    %Q registered with beta (parallelized)
                    qihat=zeros(size(ftrain));
                    gam=zeros(ntrain, length(t));
                    if REG==1
                        betahat_t = betahat(t); % Precompute for parfor
                        parfor k=1:ntrain
                            q_k = curve_to_q(ftrain(k,:));
                            gam_k = DynamicProgrammingQ_Adam(q_k, betahat_t, 0, 0);
                            gam_k(1)=0; gam_k(end)=1;
                            dumf = interp1(t, ftrain(k,:), gam_k, 'spline');
                            qihat(k,:) = curve_to_q(dumf);
                            gam(k,:) = gam_k;
                        end
                    else
                        for k=1:ntrain
                            qihat(k,:)=curve_to_q(ftrain(k,:));
                        end
                    end
                    %innerprod
                    for k=1:J
                        inp(:,k)=gap*qihat*(basis{k}(t))';
                    end
                    if any(isnan(inp(:)) | isinf(inp(:)))
                        error('NaN or Inf detected in inp matrix');
                    end
                    %beta calculation (using Huber loss for robustness)
                    Cost_fn=@(ccc) huber_loss(ytrain-ghat(ftrain(:,1))-hhat(ccc*inp'));
                    initial_cost = Cost_fn(c);
                    if isnan(initial_cost) || isinf(initial_cost)
                        error('Cost_fn is NaN/Inf at the very start!');
                    end

                    %  Choose the correct optimizer for the job ---
                    if strcmpi(FITTER_TYPE, 'poly')
                        % The cost function is SMOOTH. Use the powerful gradient-based optimizer.
                        options = optimoptions('fminunc','Display','off');
                        [cnew,val((i-1)*beta_rep+j)] = fminunc(Cost_fn,c,options);
                    else % 'pava'
                        % The cost function is NON-SMOOTH. Use the robust derivative-free optimizer.
                        options = optimset('Display','off');
                        [cnew,val((i-1)*beta_rep+j)] = fminsearch(Cost_fn,c,options);
                    end

    %                 options=optimoptions('fminunc','Display','off');
    %                 initial_cost = Cost_fn(c);
    %                 if isnan(initial_cost) || isinf(initial_cost)
    %                     fprintf('Debug: c = %s\n', mat2str(c));
    %                     fprintf('Debug: inp = %s\n', mat2str(inp));
    %                     fprintf('Debug: hhat(ccc * inp'') = %s\n', mat2str(hhat(c * inp')));
    %                 end
    %                 [cnew,val((i-1)*beta_rep+j)]=fminunc(Cost_fn,c,options);
                    c=cnew;
                end

                %s_i(beta) calculation (parallelized registration)
                yihat=ytrain';
                betahat=@(t) c(1)*basis{1}(t);
                for k=2:J
                    betahat=@(t) betahat(t)+ c(k)*basis{k}(t);
                end
                FF=figure(cross_v+10);
                clf(FF);

                % Parallel computation of qihat and gam
                nftrain = size(ftrain,1);
                qihat = zeros(nftrain, length(t));
                gam = zeros(nftrain, length(t));
                dumf_all = zeros(nftrain, length(t)); % Store for plotting

                if REG==1
                    betahat_t = betahat(t); % Precompute for parfor
                    parfor j=1:nftrain
                        q_j = curve_to_q(ftrain(j,:));
                        gam_j = DynamicProgrammingQ_Adam(q_j, betahat_t, 0, 0);
                        gam_j(1)=0; gam_j(end)=1;
                        dumf_j = interp1(t, ftrain(j,:), gam_j, 'spline');
                        qihat(j,:) = curve_to_q(dumf_j);
                        gam(j,:) = gam_j;
                        dumf_all(j,:) = dumf_j;
                    end
                else
                    for j=1:nftrain
                        qihat(j,:) = curve_to_q(ftrain(j,:));
                    end
                end

                % Sequential plotting (if clustering enabled)
                for j=1:nftrain
                    if is_Clustering>=1
                        figure(cross_v+10);
                        % 1. Capture handles for all 4 subplots immediately
                        ax1 = subplot(2,2,1); hold on;
                        ax2 = subplot(2,2,2); hold on;
                        ax3 = subplot(2,2,3); hold on;
                        ax4 = subplot(2,2,4); hold on;

                        % 3. Proceed with plotting (using handles to switch focus)
                        subplot(ax1); plot(t,betahat(t),'g','LineWidth',3);
                        subplot(ax2); plot(t,betahat(t),'g','LineWidth',3);

                        dumf = dumf_all(j,:);
                        if ytrain(j)==Class1
                            subplot(ax3); % Switch to bottom-left
                            plot(t,dumf,'Color','r');

                            subplot(ax1); % Switch back to top-left
                            plot(t,qihat(j,:),'Color','r');
                        else
                            subplot(ax4); % Switch to bottom-right
                            plot(t,dumf,'Color','b');

                            subplot(ax2); % Switch back to top-right
                            plot(t,qihat(j,:),'Color','b');
                        end

                        % 2. Link them (This is the key step)
                        linkaxes([ax1, ax2], 'y');
                        linkaxes([ax3, ax4], 'y');

                        axis([ax3, ax4], 'normal');
                        if j<=10
                            gggg=DynamicProgrammingQ_Adam(curve_to_q(f(j,:)),betahat(t),0,0);
                            ggggg=DynamicProgrammingQ_Adam(curve_to_q(f(n/2+j,:)),betahat(t),0,0);
                            figure(cross_v+20);
%                             if ytrain(j)==Class1
                                plot(t,interp1(t,f(j,:),gggg,'spline'),'Color','g','LineWidth',2);
                                hold on;
%                             else
                                plot(t,interp1(t,f(n/2+j,:),ggggg,'spline'),'Color','b','LineWidth',2);
                                hold on;
%                             end

                            figure(cross_v+30);
%                             if ytrain(j)==Class1
                                plot(t,gggg,'Color','g','LineWidth',2);
                                hold on;
%                             else
                                plot(t,ggggg,'Color','b','LineWidth',2);
                                hold on;
%                             end

                        end
                    end
                    yihat(j)=gap*betahat(t)*qihat(j,:)';
                end
                if hCENTERING==1
                    c=c/norm(yihat);
                    yihat=yihat/norm(yihat);
                end
    %             p=h_calc(yihat,ytrain',pp,h_rep);
    %             hhat=@(x) (x'.^(0:pp)*p')';   
                if is_g==0
                    [p_or_table, hhat, ~] = h_calc(yihat, ytrain', pp, h_rep, FITTER_TYPE,const);
                else
                    % --- STEP 2: Estimate hhat using residuals from g ---
                    if No_h==0
                        residuals_for_h = ytrain - ghat(ftrain(:,1));
                        [~, hhat, ~] = h_calc(yihat, residuals_for_h, pp, cross_v *100, FITTER_TYPE,const); % Use i for fig_no
                    else
                        hhat=@(x) x(:);
                    end

                    % --- STEP 3: Estimate ghat using residuals from h ---
                    residuals_for_g = ytrain - hhat(yihat);
                    predictor_for_g = ftrain(:,1);

                    % Use polyfit for a simple, non-monotonic polynomial fit
                    p_g = polyfit(predictor_for_g, residuals_for_g, pp_g);

                    % Update the ghat function handle for the next iteration
                    ghat = @(x) polyval(p_g, x);
                end
            end
            figure(888);
            subplot(2,2,cross_v);
            plot(val);
            hold on;
            plot(ones(size(val))*costtrain);
        %     plot(1:length(val),val','-',1:length(val),costtrue,'-');
        end
        if gCENTERING==1
            hhat=@(x) hhat(x)+ghat(0);
            ghat=@(x) ghat(x)-ghat(0);
        end
        %% Plotting hhat
    %     figure(1);
    %     plot(sort(x),h(sort(x)),'.-');
    %     hold on;
    %     norm(h(x)'-hhat(x))/norm(h(x))

        %% Plotting beta
    %     figure(2);
    % %     plot(t,beta(t)/norm(beta(t)),'.');
    % %     hold on;
    %     gtemp=DynamicProgrammingQ_Adam(curve_to_q(betahat(t)),curve_to_q(beta(t)),0,0);
    %     
    %     plot(t,interp1(t,betahat(t),gtemp),'-');
    %     hold on;
    %     norm(curve_to_q(interp1(t,betahat(t),gtemp))-curve_to_q(beta(t)/norm(beta(t))))

        %% Plotting ghat
    %     figure(3);
    %     plot(sort(f(:,1)),g(sort(f(:,1))),'.-');
    %     hold on;
    %     norm(g(f(:,1))-ghat(f(:,1)))/norm(g(f(:,1)))

        % Training Output
        betahat(t);
        hhat(yihat);
        ytr_pred=ghat(ftrain(:,1))+hhat(yihat);
        if is_Clustering>=1
            for k=1:size(ftrain,1)
                if abs(ytr_pred(k)-Class1)<abs(ytr_pred(k)-Class2)
                    ytr_pred(k)=Class1;
                else
                    ytr_pred(k)=Class2;
                end
            end
        end
        trainR2(cross_v)=1-trimmean((ytrain-ytr_pred).^2,10)/trimmean((mean(ytrain)*ones(length(ytrain),1)-ytrain).^2,10);
        figure(cross_v+4);%777);
%         subplot(2,2,cross_v);
        plot(ytrain',ytr_pred,'.',ytrain',ytrain','-');
        hold on;

        %Testing (parallelized)
        nftest = size(ftest,1);
        qihat2 = zeros(nftest, length(t));
        yihat2 = zeros(1, nftest);

        if REG==1
            betahat_t = betahat(t); % Precompute for parfor
            parfor j=1:nftest
                q_j = curve_to_q(ftest(j,:));
                gam_j = DynamicProgrammingQ_Adam(q_j, betahat_t, 0, 0);
                gam_j(1)=0; gam_j(end)=1;
                dumf = interp1(t, ftest(j,:), gam_j, 'spline');
                qihat2(j,:) = curve_to_q(dumf);
            end
        else
            for j=1:nftest
                qihat2(j,:) = curve_to_q(ftest(j,:));
            end
        end
        % Compute yihat2 (sequential - fast dot products)
        betahat_t = betahat(t);
        for j=1:nftest
            yihat2(j) = gap*betahat_t*qihat2(j,:)';
        end
        if hCENTERING==1
            yihat2=yihat2/norm(yihat);%from training
        end
        yte_pred(:,cross_v)=ghat(ftest(:,1))+hhat(yihat2);
        if is_Clustering>=1
            for k=1:size(ftest,1)
%                 yte_pred(k,cross_v)=(abs(yte_pred(k,cross_v)-Class1)<abs(yte_pred(k,cross_v)))*Class1;
                if abs(yte_pred(k,cross_v)-Class1)<abs(yte_pred(k,cross_v)-Class2)
                    yte_pred(k,cross_v)=Class1;
                else
                    yte_pred(k,cross_v)=Class2;
                end
            end
        end
        ytest_store(:,cross_v)=ytest;
        testR2(cross_v)=1-trimmean((ytest-yte_pred(:,cross_v)).^2,10)/trimmean((mean(ytrain)*ones(length(ytest),1)-ytest).^2,10);
        figure(cross_v+4);%777);
%         subplot(2,2,cross_v);
        if is_Clustering>=1
            plot(1:size(ftest,1),ytest','.',1:size(ftest,1),yte_pred(:,cross_v));
        end
        plot(ytest',yte_pred(:,cross_v),'.');
        betamat((ci-1)*4+cross_v,:)=betahat(t);
        hmat((ci-1)*4+cross_v,:)=hhat(ht)';
        gmat((ci-1)*4+cross_v,:)=ghat(gt);

        % Store results for this (n, cross_v) combination
        row_idx = row_idx + 1;
        results_table(row_idx, :) = [n, cross_v, trainR2(cross_v), testR2(cross_v)];
    end
    trainR2
    testR2
end
    end % end repeat loop
    % --- Save intermediate results ---
    save('partial_results_codehelp.mat', 'results_table', 's_idx', 'n_vals');
    fprintf('Saved intermediate results (s_idx=%d).\n', s_idx);
end  % end sample size loop

%% Display Final Results Table
fprintf('\n============================================================\n');
fprintf('              R^2 Comparison Table (Detailed)\n');
fprintf('============================================================\n');
fprintf(' %-10s | %-8s | %-12s | %-12s \n', 'n', 'cross_v', 'Train R2', 'Test R2');
fprintf('------------------------------------------------------------\n');
for i = 1:size(results_table, 1)
    fprintf(' %-10d | %-8d | %-12.4f | %-12.4f \n', ...
        results_table(i,1), results_table(i,2), results_table(i,3), results_table(i,4));
end
fprintf('============================================================\n');

% Compute median test R^2 per sample size
median_testR2 = zeros(1, n_sample_sizes);
for i = 1:n_sample_sizes
    idx = results_table(:,1) == n_vals(i);
    median_testR2(i) = median(results_table(idx, 4));
end
figure(6769);
hold on;
plot(n_vals, median_testR2, 'o-', 'Color', [0 0.4470 0.7410], ...
    'LineWidth', 2.0, 'MarkerSize', 10, 'DisplayName', 'SI-ScoSH');
xlabel('Number of Observations (n)', 'FontSize', 14);
ylabel('Cross Validated Test set Prediction R^2', 'FontSize', 14);
title('$\sigma =0.5$', 'FontSize', 14, 'Interpreter', 'latex');
legend('FontSize', 14, 'Location', 'best');
ax = gca; ax.FontSize = 14;
box on;
% %% CI Prints
% for i=1:(n_ci*4)
%      gtemp=DynamicProgrammingQ_Adam(curve_to_q(betamat(i,:)),curve_to_q(betamat(1,:)),0,0);
%     %     
%     %     plot(t,interp1(t,betahat(t),gtemp),'-');
%     figure(1);
%     hold on;
%     plot(t,interp1(t,betamat(i,:),gtemp),'-');
% %     text(t(10),betamat(i,10),cellstr(num2str(i)))
%     figure(2);
%     hold on;
% %     plot(sort(f(:,1)),gmat{i}(sort(f(:,1)))*const,'.-');
%     plot(gt,gmat*const,'.-');
%     figure(3);
%     hold on;
% %     plot(sort(x),hmat{i}(sort(x))*const,'.-');
%     plot(ht,hmat*const,'.-');
% end
% 
% %% BETA CI 
% betamatstore=betamat;
% manipidx=setdiff(1:size(betamat,1),[]);
% betamat=betamat(manipidx,:);
% lowbd=zeros(1,size(t,2));
% uppbd=zeros(1,size(t,2));
% for tp=1:size(t,2)
%     vect=betamat(:,tp);
%     vect=sort(vect);
%     lowbd(tp)=vect(max(2,round(0.05*size(betamat,1))));
%     uppbd(tp)=vect(min(size(betamat,1)-1,round(0.95*size(betamat,1))));
% end
% figure(11);
% patch([t flip(t)],[lowbd flip(uppbd)],[1 1 1]*0.8,'EdgeColor','none');
% hold on;
% plot(t,lowbd,'-',t,uppbd,'-');
% 
% %% h CI
% hmatstore=hmat;
% manipidx=setdiff(1:size(hmat,1),[]);
% hmat=hmat(manipidx,:);
% lowbd=zeros(1,size(ht,2));
% uppbd=zeros(1,size(ht,2));
% for tp=1:size(ht,2)
%     vect=hmat(:,tp);
% %     vect=sort(vect);
%     vect=sort(vect(~isnan(vect)));
%     lowbd(tp)=vect(max(2,ceil(0.05*size(hmat,1))));%max(1,...
%     uppbd(tp)=vect(min(size(vect,1)-1,floor(0.95*size(hmat,1))));%t,1)-0,...
% end
% figure(22);
% patch([ht flip(ht)],[lowbd flip(uppbd)],[1 1 1]*0.8,'EdgeColor','none');
% hold on;
% plot(ht,lowbd,'-',ht,uppbd,'-');
% 
% %% g CI
% gmatstore=gmat;
% manipidx=setdiff(1:size(gmat,1),[]);
% gmat=gmat(manipidx,:);
% lowbd=zeros(1,size(gt,2));
% uppbd=zeros(1,size(gt,2));
% for tp=1:size(gt,2)
%     vect=gmat(:,tp);
% %     vect=sort(vect);
%     vect=sort(vect(~isnan(vect)));
%     lowbd(tp)=vect(max(2,ceil(0.05*size(gmat,1))));%max(1,...
%     uppbd(tp)=vect(min(size(vect,1)-1,floor(0.95*size(gmat,1))));%t,1)-0,...
% end
% figure(22);
% patch([gt flip(gt)],[lowbd flip(uppbd)],[1 1 1]*0.8,'EdgeColor','none');
% hold on;
% plot(gt,lowbd,'-',gt,uppbd,'-');
