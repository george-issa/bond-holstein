clear;
clc;
close all;

%set up the figure
figure('Renderer', 'painters', 'Position', [10 10 600 500])
set(gcf,'color','white')

%define panel locations and smoothing parameter
p = 0.999;

Tc_bond = 1./1.58;   y = [-100:0.1:100]; x = ones(size(y));

% panel b
A = dlmread('data/holstein_energy.csv');
T = 1./A(:,1);
dV_by_dT = gradient(A(:,2))./gradient(T);
d2V_by_dT2 = gradient(dV_by_dT)./gradient(T);
plot(T,d2V_by_dT2,'bo','MarkerFaceColor','b','MarkerSize',8); hold on;

set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[0.25:0.25:2],'YTick',[-100:10:100],'LineWidth',1)
xtickangle(0)
plot(Tc_bond*x, y,'--k','HandleVisibility','off','LineWidth',1)

TT = [0.25:0.01:2];
values = csaps(T,A(:,2),p,TT); 
dFit_by_dT = gradient(values)./gradient(TT);
d2Fit_by_dT2 = gradient(dFit_by_dT)./gradient(TT);
plot(TT, d2Fit_by_dT2,'-b','HandleVisibility','off','LineWidth',1)


set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[0:0.5:2],'YTick',[-20:10:30],'LineWidth',1)
ylabel('$d^2\mathcal{V}/dT^2$','FontSize',25,'Interpreter','latex')
xlabel('$T/t$','FontSize',25,'Interpreter','latex')

axis([0.25,2,-20,30])
