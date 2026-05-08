clear;
clc;
close all;

Left = 0.05;
Bottom = 0.15;
Height = 0.3;
Width = 0.17;
voffset = Height*1.42;
hoffset = Width*1.05;
fontsize = 26;

% Load the data
A = load('data/charge_density_real_space.csv');

%set up the figure
figure('Renderer', 'painters', 'Position', [10 10 600 350])
set(gcf,'color','white')

% beta = 1;

subplot('position',[Left+0*hoffset,Bottom+voffset,Width,Height]); box on; hold on;
beta = 1; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$\beta t = 1$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+1*hoffset,Bottom+voffset,Width,Height]); box on; hold on;
beta = 1.1; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.1$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+2*hoffset,Bottom+voffset,Width,Height]); box on; hold on;
beta = 1.2; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.2$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+3*hoffset,Bottom+voffset,Width,Height]); box on; hold on;
beta = 1.3; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.3$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+4*hoffset,Bottom+voffset,Width,Height]); box on; hold on;
beta = 1.4; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.4$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+0*hoffset,Bottom+0*voffset,Width,Height]); box on; hold on;
beta = 1.5; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.5$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+1*hoffset,Bottom+0*voffset,Width,Height]); box on; hold on;
beta = 1.6; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.6$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+2*hoffset,Bottom+0*voffset,Width,Height]); box on; hold on;
beta = 1.7; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.7$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')
colorbar('southoutside','FontName','Times','FontSize',fontsize)

subplot('position',[Left+3*hoffset,Bottom+0*voffset,Width,Height]); box on; hold on;
beta = 1.8; load_data;
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.8$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

subplot('position',[Left+4*hoffset,Bottom+0*voffset,Width,Height]); box on; hold on;
beta = 1.9; load_data; 
surf(x,y,z); axis([0,8,0,8]); colormap('jet'); caxis([0,2])
set(gca,'LineWidth',1,'XTick',[-2:1:-2],'YTick',[-2:1:-2])
title('$1.9$','FontSize',fontsize,'FontWeight','normal','FontName','Times','Interpreter','latex')

saveas(gcf,'../figures/figure1.png','png')
