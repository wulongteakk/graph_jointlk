

import React, { useState } from 'react';
import { VisualizationData } from '../../services/QnaAPI'; // 导入类型

interface Props {
  data: VisualizationData;
  onClose: () => void;
}

// --- 子组件：Token-实体对齐可视化 ---
const TokenEntityAlignment: React.FC<{ alignmentData: VisualizationData['token_entity_alignment'] }> = ({ alignmentData }) => {
  const { tokens, entities, alignment_matrix } = alignmentData;
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(null);
  const [hoveredEntityIndex, setHoveredEntityIndex] = useState<number | null>(null);

  // 阈值，低于此值不显示连线或设为很淡
  const alignmentThreshold = 0.1;

  return (
    <div className="my-4 p-4 border rounded-lg bg-gray-50 dark:bg-gray-800">
      <h3 className="text-lg font-semibold mb-3 dark:text-white">Token-Entity Alignment (Layer 1)</h3>
      <div className="flex justify-between relative">
        {/* 左侧：Tokens */}
        <div className="w-1/3 pr-4">
          <h4 className="text-md font-medium mb-2 dark:text-gray-300">Input Tokens</h4>
          {tokens.map((token, tokenIndex) => (
            <div
              key={tokenIndex}
              className={`p-1 my-1 rounded cursor-pointer transition-all ${hoveredTokenIndex === tokenIndex ? 'bg-blue-200 dark:bg-blue-700 font-bold' : 'bg-gray-100 dark:bg-gray-700'}`}
              onMouseEnter={() => setHoveredTokenIndex(tokenIndex)}
              onMouseLeave={() => setHoveredTokenIndex(null)}
            >
              <span className="text-sm dark:text-gray-100">{token}</span>
            </div>
          ))}
        </div>

        {/* 右侧：Entities */}
        <div className="w-1/3 pl-4 text-right">
          <h4 className="text-md font-medium mb-2 dark:text-gray-300">Graph Entities</h4>
          {entities.map((entity, entityIndex) => (
            <div
              key={entity.id}
              className={`p-1 my-1 rounded cursor-pointer transition-all ${hoveredEntityIndex === entityIndex ? 'bg-purple-200 dark:bg-purple-700 font-bold' : 'bg-gray-100 dark:bg-gray-700'}`}
              onMouseEnter={() => setHoveredEntityIndex(entityIndex)}
              onMouseLeave={() => setHoveredEntityIndex(null)}
            >
              <span className="text-sm dark:text-gray-100">{entity.label} ({entity.type})</span>
            </div>
          ))}
        </div>

        {/* 中间：对齐线 (使用SVG绘制) */}
        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
          {tokens.map((_, tokenIndex) =>
            entities.map((_, entityIndex) => {
              const score = alignment_matrix[tokenIndex][entityIndex];
              if (score < alignmentThreshold) return null;

              // 计算连线高亮状态
              const highlighted = hoveredTokenIndex === tokenIndex || hoveredEntityIndex === entityIndex;

              return (
                <line
                  key={`${tokenIndex}-${entityIndex}`}
                  x1="33%" // 左侧列表结束位置
                  y1={tokenIndex * 36 + 40} // 估算的y坐标 (需要根据实际元素大小调整)
                  x2="66%" // 右侧列表开始位置
                  y2={entityIndex * 36 + 40} // 估算的y坐标
                  stroke={highlighted ? 'rgba(255, 0, 0, 0.9)' : 'rgba(0, 100, 255, 0.4)'}
                  strokeWidth={highlighted ? 2.5 : score * 3} // 根据分数调整粗细
                  strokeOpacity={highlighted ? 1 : 0.5 + score * 0.5}
                />
              );
            })
          )}
        </svg>
      </div>
    </div>
  );
};

// --- 子组件：图剪裁对比 ---
const GraphPruningComparison: React.FC<{ pruningData: VisualizationData['graph_pruning'] }> = ({ pruningData }) => {
    // 这里需要一个图可视化库 (如 vis-network, react-flow, or d3) 来渲染 'before' 和 'after' 图。
    // 为简化起见，我们仅以文本形式展示节点和分数。
    const { before, after } = pruningData;
    return (
        <div className="my-4 p-4 border rounded-lg bg-gray-50 dark:bg-gray-800">
            <h3 className="text-lg font-semibold mb-3 dark:text-white">Graph Pruning Comparison</h3>
            <div className="flex space-x-4">
                <div className="w-1/2">
                    <h4 className="font-medium dark:text-gray-200">Before Pruning ({before.nodes.length} nodes)</h4>
                    <ul className="list-disc pl-5 dark:text-gray-300">
                        {before.nodes.sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0)).map(node => (
                            <li key={node.id} title={`ID: ${node.id}`}>
                                {node.label} (Score: {node.importance?.toFixed(3)})
                            </li>
                        ))}
                    </ul>
                </div>
                <div className="w-1/2">
                    <h4 className="font-medium dark:text-gray-200">After Pruning ({after.nodes.length} nodes)</h4>
                     <ul className="list-disc pl-5 dark:text-gray-300">
                        {after.nodes.map(node => (
                            <li key={node.id}>{node.label}</li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    );
};


// --- 主模态框组件 ---
export const JointLKInferenceVisualizer: React.FC<Props> = ({ data, onClose }) => {
  const [activeTab, setActiveTab] = useState<'alignment' | 'pruning'>('alignment');

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto p-6">
        <div className="flex justify-between items-center border-b pb-2 dark:border-gray-700">
          <h2 className="text-xl font-bold dark:text-white">JointLK Inference Process Visualization</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800 dark:hover:text-gray-200">&times;</button>
        </div>

        {/* 可视化内容切换 */}
        <div className="my-4 border-b dark:border-gray-700">
            <button className={`py-2 px-4 ${activeTab === 'alignment' ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400' : 'text-gray-500'}`} onClick={() => setActiveTab('alignment')}>Token-Entity Alignment</button>
            <button className={`py-2 px-4 ${activeTab === 'pruning' ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400' : 'text-gray-500'}`} onClick={() => setActiveTab('pruning')}>Graph Pruning</button>
        </div>

        {/* 内容区域 */}
        <div>
            {activeTab === 'alignment' && <TokenEntityAlignment alignmentData={data.token_entity_alignment} />}
            {activeTab === 'pruning' && <GraphPruningComparison pruningData={data.graph_pruning} />}
        </div>

        {/* 其他信息 */}
        <div className="text-right text-sm text-gray-600 dark:text-gray-400 mt-4">
            Confidence Score: {(data.confidence_score * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
};