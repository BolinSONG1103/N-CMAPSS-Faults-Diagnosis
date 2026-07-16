classdef thesis_plot
    %THESIS_PLOT 第3章统一论文制图规范与导出入口。
    % 采用色盲友好的 Okabe-Ito 配色、紧凑无边框坐标轴，并同时输出
    % 600 dpi PNG、矢量 PDF 和 SVG，便于论文排版与后期无损编辑。

    methods (Static)
        function c=colors()
            c=[0 114 178; 213 94 0; 0 158 115; 230 159 0; ...
                86 180 233; 204 121 167; 0 0 0; 240 228 66]/255;
        end

        function c=semantic_colors()
            % 全章固定方法配色；同一方法在所有正文图中保持同一颜色。
            c=struct( ...
                'D',[0 114 178]/255, ...
                'B',[213 94 0]/255, ...
                'Z_zero',[0 0 0]/255, ...
                'E_Lin',[230 159 0]/255, ...
                'E_MLP',[0 158 115]/255, ...
                'E_MixLinear',[204 121 167]/255, ...
                'E_WPMixer',[86 180 233]/255, ...
                'truth',[.28 .28 .28], ...
                'secondary',[.55 .55 .55]);
        end

        function map=viridis(n)
            if nargin<1, n=256; end
            anchors=[68 1 84;59 82 139;33 145 140;94 201 98;253 231 37]/255;
            map=interp1(linspace(0,1,size(anchors,1)),anchors,linspace(0,1,n),'pchip');
            map=min(max(map,0),1);
        end

        function map=diverging(n)
            if nargin<1, n=257; end
            lo=[49 54 149]/255; mid=[247 247 247]/255; hi=[165 0 38]/255;
            k=floor(n/2);
            map=[interp1([1 k+1],[lo;mid],1:k+1); ...
                interp1([1 n-k],[mid;hi],2:n-k)];
        end

        function f=new(widthCm,heightCm)
            if nargin<1, widthCm=17.8; end
            if nargin<2, heightCm=10.5; end
            f=figure('Visible','off','Color','white','Units','centimeters', ...
                'Position',[2 2 widthCm heightCm],'PaperPositionMode','auto', ...
                'InvertHardcopy','off');
        end

        function apply(f)
            axesList=findall(f,'Type','axes');
            for i=1:numel(axesList)
                ax=axesList(i);
                set(ax,'FontName','Arial','FontSize',8,'LineWidth',0.75, ...
                    'TickDir','out','TickLength',[.015 .015],'Box','off', ...
                    'Layer','top','Color','none','XColor',[.18 .18 .18], ...
                    'YColor',[.18 .18 .18],'GridColor',[.82 .82 .82], ...
                    'GridAlpha',.55,'MinorGridAlpha',.25, ...
                    'XMinorGrid','off','YMinorGrid','off');
            end
            lines=findall(f,'Type','line');
            for i=1:numel(lines)
                if strcmp(lines(i).LineStyle,'none')
                    lines(i).MarkerSize=5;
                elseif lines(i).LineWidth<1.2
                    lines(i).LineWidth=1.35;
                end
            end
            legends=findall(f,'Type','legend');
            for i=1:numel(legends)
                set(legends(i),'Box','off','FontName','Arial','FontSize',7.5);
            end
            textItems=findall(f,'Type','text');
            for i=1:numel(textItems)
                if isempty(textItems(i).FontName) || strcmpi(textItems(i).FontName,'Helvetica')
                    textItems(i).FontName='Arial';
                end
            end
        end

        function panel(ax,label)
            text(ax,-.055,1.015,['(' label ')'],'Units','normalized','FontName','Arial', ...
                'FontSize',9.5,'FontWeight','bold','HorizontalAlignment','left', ...
                'VerticalAlignment','bottom','Clipping','off');
        end

        function paths=export(f,outDir,stem,includeSvg)
            if nargin<4, includeSvg=true; end
            if ~isfolder(outDir), mkdir(outDir); end
            thesis_plot.apply(f);
            drawnow;
            pngPath=fullfile(outDir,[stem '.png']);
            pdfPath=fullfile(outDir,[stem '.pdf']);
            svgPath='';
            exportgraphics(f,pngPath,'Resolution',600,'BackgroundColor','white');
            exportgraphics(f,pdfPath,'ContentType','vector','BackgroundColor','white');
            if includeSvg
                svgPath=fullfile(outDir,[stem '.svg']);
                try
                    exportgraphics(f,svgPath,'ContentType','vector','BackgroundColor','white');
                catch
                    svgPath='';
                end
            end
            paths=struct('png',pngPath,'pdf',pdfPath,'svg',svgPath);
        end
    end
end
