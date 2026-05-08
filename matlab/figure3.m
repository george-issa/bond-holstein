clear;
clc;
close all;

%set up the figure
figure('Renderer', 'painters', 'Position', [10 10 600 900])
set(gcf,'color','white')

%define panel locations and smoothing parameter
Left = 0.15;
Bottom = 0.09;
Height = 0.29;
Width = 0.8;
voffset = 0.3;
p = 1;

Tc_bond = 1./1.58;   y = [-10:0.1:100]; x = ones(size(y));

%panel a
A = dlmread('data/hopping_energy.csv')
T = 1./A(:,1);
dK_by_dT = gradient(A(:,2))./gradient(T);

subplot('position',[Left,Bottom+2*voffset,Width,Height]); box on; hold on;
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[3.4:0.4:4.4],'YTick',[0:20:100],'LineWidth',1)
xtickangle(0)
errorbar(T,A(:,2),A(:,3),'bo','MarkerFaceColor','b','MarkerSize',8)
plot(Tc_bond*x, y,'--k','HandleVisibility','off')
s0 = csaps(T,A(:,2),p); fnplt(s0,'-b');
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[0:0.25:2],'YTick',[-1.1:0.1:-0.6],'LineWidth',1)
ylabel('$\mathcal{K}/t$','FontSize',25,'Interpreter','latex')
axis([0.25,2,-1.15,-0.7])
xticklabels({' ',' ',' ',' ',' ',' ',' ',' '})
text(0.29,-0.735,'(a)','FontSize',25,'FontName','Times')

axes('Position',[.28 .835 .3 .12],'LineWidth',1); hold on; box on;
plot(T,dK_by_dT,'bo','MarkerFaceColor','b','MarkerSize',6)
plot(Tc_bond*x, y,'--k','HandleVisibility','off')
s0 = csaps(T,dK_by_dT,0.99995); fnplt(s0,'-b');
set(gca,'FontSize',20,'FontName','Times',...
        'Xtick',[0.25:0.25:2],'YTick',[-1:0.2:2],'LineWidth',1)
xtickangle(0)
xticklabels({' ','0.5',' ','1',' ','1.5',' ','2'})
xlabel('$T/t$','FontSize',20,'Interpreter','latex')
ylabel('$d\mathcal{K}/dT$','FontSize',20,'Interpreter','latex')
axis([0.25,2,-0.25,0.4])

% panel b
A = dlmread('data/holstein_energy.csv')
T = 1./A(:,1);
dV_by_dT = gradient(A(:,2))./gradient(T);

subplot('position',[Left,Bottom+1*voffset,Width,Height]); box on; hold on;
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[3.4:0.4:4.4],'YTick',[0:20:100],'LineWidth',1)
xtickangle(0)
errorbar(T,A(:,2),A(:,3),'bo','MarkerFaceColor','b','MarkerSize',8)
plot(Tc_bond*x, y,'--k','HandleVisibility','off')
s0 = csaps(T,A(:,2),p); fnplt(s0,'-b');
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[0:0.25:2],'YTick',[-2.0:0.4:-0.8],'LineWidth',1)
ylabel('$\mathcal{V}/t$','FontSize',25,'Interpreter','latex')
axis([0.25,2,-2.4,-0.8])
xticklabels({' ',' ',' ',' ',' ',' ',' ',' '})
text(0.29,-0.92,'(b)','FontSize',25,'FontName','Times')

axes('Position',[.52 .46 .4 .14],'LineWidth',1); hold on; box on;
plot(T,dV_by_dT,'bs','MarkerFaceColor','b','MarkerSize',6)
plot(Tc_bond*x, y,'--k','HandleVisibility','off')
s0 = csaps(T,dV_by_dT,0.99995); fnplt(s0,'-b');
set(gca,'FontSize',20,'FontName','Times',...
        'Xtick',[0.25:0.25:2],'YTick',[-1:1:4],'LineWidth',1)
xtickangle(0)
xticklabels({' ','0.5',' ','1',' ','1.5',' ','2'})
xlabel('$T/t$','FontSize',20,'Interpreter','latex')
ylabel('$d\mathcal{V}/dT$','FontSize',20,'Interpreter','latex')
axis([0.25,2,0,4])


% Plot the double occupancy in panel (c)
A = dlmread('data/double_occ.csv');
T = 1./A(:,1);
dD_by_dT = gradient(A(:,2))./gradient(T);

subplot('position',[Left,Bottom+0*voffset,Width,Height]); box on; hold on;
errorbar(T,A(:,2),A(:,3),'bo','MarkerFaceColor','b','MarkerSize',8)
plot(Tc_bond*x, y,'--k','HandleVisibility','off')
s0 = csaps(T,A(:,2),p); fnplt(s0,'-b');
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[0.25:0.25:2],'YTick',[0.28:0.04:0.5],'LineWidth',1)
xlabel('$T/t$','FontSize',25,'Interpreter','latex')
ylabel('$\mathcal{D}$','FontSize',25,'Interpreter','latex')
axis([0.25,2,0.28,0.45])
xticklabels({' ','0.5',' ','1',' ','1.5',' ','2'})
text(0.29,0.4375,'(c)','FontSize',25,'FontName','Times')

axes('Position',[.52 .22 .4 .14],'LineWidth',1); hold on; box on;
plot(T,dD_by_dT,'bs','MarkerFaceColor','b','MarkerSize',6)
plot(Tc_bond*x, y,'--k','HandleVisibility','off')
s0 = csaps(T,dD_by_dT,0.99995); fnplt(s0,'-b');
set(gca,'FontSize',20,'FontName','Times',...
        'Xtick',[0.25:0.25:2],'YTick',[-1:0.1:0],'LineWidth',1)
xtickangle(0)
xticklabels({' ','0.5',' ','1',' ','1.5',' ','2'})
xlabel('$T/t$','FontSize',20,'Interpreter','latex')
ylabel('$d\mathcal{D}/dT$','FontSize',20,'Interpreter','latex')
axis([0.25,2,-0.4,0])

saveas(gcf,'../figures/figure3.pdf','pdf')

