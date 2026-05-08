clear;
clc;
close all;

% Load the data
site_L8 = load('data/site_L8.csv');
site_L10 = load('data/site_L10.csv');
site_L12 = load('data/site_L12.csv');
bond_L8 = load('data/sorted_bond_a0.6325_L8.csv');
bond_L10 = load('data/sorted_bond_a0.6325_L10.csv');
bond_L12 = load('data/sorted_bond_a0.6325_L12.csv');

bond_L8(:,2:3) = bond_L8(:,2:3);
bond_L10(:,2:3) = bond_L10(:,2:3);
bond_L12(:,2:3) = bond_L12(:,2:3);

%set up the figure
figure('Renderer', 'painters', 'Position', [10 10 600 500])
set(gcf,'color','white')

%define panel locations and smoothing parameter
Left = 0.15;
Bottom = 0.14;
Height = 0.4;
Width = 0.4;
voffset = 0.42;
hoffset = 0.42;
p = 1;

Tc_site = 4.035; 
Tc_bond = 1.58;
y = [0:0.1:100]; x = ones(size(y));

%panel a
subplot('position',[Left,Bottom+voffset,Width,Height]); box on; hold on;
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[3.4:0.4:4.4],'YTick',[0:20:100],'LineWidth',1)
xtickangle(0)
errorbar(site_L8(:,1),site_L8(:,2),site_L8(:,3),'rs','MarkerFaceColor','r','MarkerSize',8)
errorbar(site_L10(:,1),site_L10(:,2),site_L10(:,3),'bo','MarkerFaceColor','b','MarkerSize',8)
errorbar(site_L12(:,1),site_L12(:,2),site_L12(:,3),'gd','MarkerFaceColor','g','MarkerSize',8)
s0 = csaps(site_L8(:,1),site_L8(:,2),1); fnplt(s0,'-r');
s0 = csaps(site_L10(:,1),site_L10(:,2),1); fnplt(s0,'-b');
s0 = csaps(site_L12(:,1),site_L12(:,2),1); fnplt(s0,'-g');
axis([3.4,4.6,0,100])
xticklabels({' ',' ',' ',' ',' ',' '})
ylabel('$S(\pi,\pi)$','FontSize',25,'Interpreter','latex')
text(3.45,100*0.9,'(a)','FontSize',25,'FontName','Times')

% panel c
subplot('position',[Left+hoffset,Bottom+voffset,Width,Height]); box on; hold on; 
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[0:0.5:3],'YTick',[0:20:100],'LineWidth',1)
xtickangle(0)
errorbar(bond_L8(:,1),bond_L8(:,2),bond_L8(:,3),'rs','MarkerFaceColor','r','MarkerSize',8)
errorbar(bond_L10(:,1),bond_L10(:,2),bond_L10(:,3),'bo','MarkerFaceColor','b','MarkerSize',8)
errorbar(bond_L12(:,1),bond_L12(:,2),bond_L12(:,3),'gd','MarkerFaceColor','g','MarkerSize',8)
s0 = csaps(bond_L8(:,1),bond_L8(:,2),1); fnplt(s0,'-r');
s0 = csaps(bond_L10(:,1),bond_L10(:,2),1); fnplt(s0,'-b');
s0 = csaps(bond_L12(:,1),bond_L12(:,2),1); fnplt(s0,'-g');
axis([0.75,2.5,0,100])
xticklabels({' ',' ',' ',' ',' ',' '})
yticklabels({' ',' ',' ',' ',' ',' '})
text(0.8,100*0.9,'(c)','FontSize',25,'FontName','Times')


% Rescale the data
site_L8(:,2:3) = site_L8(:,2:3)*(8^(-7/4));
site_L10(:,2:3) = site_L10(:,2:3)*(10^(-7/4));
site_L12(:,2:3) = site_L12(:,2:3)*(12^(-7/4));

bond_L8(:,2:3) = bond_L8(:,2:3)*(8^(-7/4));
bond_L10(:,2:3) = bond_L10(:,2:3)*(10^(-7/4));
bond_L12(:,2:3) = bond_L12(:,2:3)*(12^(-7/4));

%panel b
subplot('position',[Left,Bottom,Width,Height]); box on; hold on; 
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[3.4:0.4:4.4],'YTick',[0:0.25:1],'LineWidth',1)
xtickangle(0)
errorbar(site_L8(:,1),site_L8(:,2),site_L8(:,3),'rs','MarkerFaceColor','r','MarkerSize',8)
errorbar(site_L10(:,1),site_L10(:,2),site_L10(:,3),'bo','MarkerFaceColor','b','MarkerSize',8)
errorbar(site_L12(:,1),site_L12(:,2),site_L12(:,3),'gd','MarkerFaceColor','g','MarkerSize',8)
s0 = csaps(site_L8(:,1),site_L8(:,2),1); fnplt(s0,'-r');
s0 = csaps(site_L10(:,1),site_L10(:,2),1); fnplt(s0,'-b');
s0 = csaps(site_L12(:,1),site_L12(:,2),1); fnplt(s0,'-g');
plot(Tc_site*x, y,'--k','HandleVisibility','off')
axis([3.4,4.6,0,1.25])
ylabel('$S(\pi,\pi)/L^{7/4}$','FontSize',25,'Interpreter','latex')
xlabel('$\beta t$','FontSize',25,'Interpreter','latex')
text(3.45,1.25*0.9,'(b)','FontSize',25,'FontName','Times')

%panel d
subplot('position',[Left+hoffset,Bottom,Width,Height]); box on; hold on; 
set(gca,'FontSize',25,'FontName','Times',...
        'Xtick',[0:0.5:3],'YTick',[0:0.25:1],'LineWidth',1)
xtickangle(0)
errorbar(bond_L8(:,1),bond_L8(:,2),bond_L8(:,3),'rs','MarkerFaceColor','r','MarkerSize',8)
errorbar(bond_L10(:,1),bond_L10(:,2),bond_L10(:,3),'bo','MarkerFaceColor','b','MarkerSize',8)
errorbar(bond_L12(:,1),bond_L12(:,2),bond_L12(:,3),'gd','MarkerFaceColor','g','MarkerSize',8)
s0 = csaps(bond_L8(:,1),bond_L8(:,2),1); fnplt(s0,'-r');
s0 = csaps(bond_L10(:,1),bond_L10(:,2),1); fnplt(s0,'-b');
s0 = csaps(bond_L12(:,1),bond_L12(:,2),1); fnplt(s0,'-g');
plot(Tc_bond*x, y,'--k','HandleVisibility','off')
axis([0.75,2.5,0,1.25])
yticklabels({' ',' ',' ',' ',' ',' '})
xlabel('$\beta t$','FontSize',25,'Interpreter','latex')
text(0.8,1.25*0.9,'(d)','FontSize',25,'FontName','Times')
legend('$8\times 8$','$10\times 10$','$12\times 12$')
legend('location','southeast','Interpreter','latex')
legend boxoff;

saveas(gcf,'../figures/figure2.pdf','pdf')
