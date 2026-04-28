// JointLKViewModal.tsx

import { Banner, Dialog, Flex, IconButtonArray, LoadingSpinner } from '@neo4j-ndl/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { GraphType, GraphViewModalProps, OptionType, Scheme, UserCredentials } from '../../types';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import NVL from '@neo4j-nvl/base';
import type { Node, Relationship } from '@neo4j-nvl/base';
import { Resizable } from 're-resizable';
import {
  ArrowPathIconOutline,
  DragIcon,
  FitToScreenIcon,
  MagnifyingGlassMinusIconOutline,
  MagnifyingGlassPlusIconOutline,
} from '@neo4j-ndl/react/icons';
// 1. 引入 recharts
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import IconButtonWithToolTip from '../UI/IconButtonToolTip';
// 8. 从 Utils 中移除 filterData
import { processGraphData } from '../../utils/Utils';
import { useCredentials } from '../../context/UserCredentials';
import { LegendsChip } from './LegendsChip';
import graphQueryAPI from '../../services/GraphQuery';
import {
  // 8. 移除不再需要的常量
  // entityGraph,
  graphQuery,
  // JointlkView, // 将被 newOptions 替代
  // intitalGraphType,
  // knowledgeGraph,
  // lexicalGraph,
  mouseEventCallbacks,
  nvlOptions, // 仍将作为基础配置
  queryMap,
} from '../../utils/Constants';
import DropdownComponent from '../Dropdown';

// 2. 更新 Props 类型，加入剪枝前的数据
interface PruningGraphViewModalProps extends GraphViewModalProps {
  unprunedNodes?: Node[];
  unprunedRelationships?: Relationship[];
}

const JointlkViewModal: React.FunctionComponent<PruningGraphViewModalProps> = ({
  open,
  inspectedName,
  setGraphViewOpen,
  viewPoint,
  nodeValues,
  relationshipValues,
  selectedRows,
  // 2. 解构新的 props
  unprunedNodes,
  unprunedRelationships,
}) => {
  const nvlRef = useRef<NVL>(null);
  const [nodes, setNodes] = useState<Node[]>([]); // 剪枝后的节点
  const [relationships, setRelationships] = useState<Relationship[]>([]); // 剪枝后的关系
  // 8. 移除 graphType 状态
  // const [graphType, setGraphType] = useState<GraphType[]>(intitalGraphType);
  const [allNodes, setAllNodes] = useState<Node[]>([]); // 剪枝后的所有节点 (用于图例)
  const [allRelationships, setAllRelationships] = useState<Relationship[]>([]); // 剪枝后的所有关系
  const [loading, setLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<'unknown' | 'success' | 'danger'>('unknown');
  const [statusMessage, setStatusMessage] = useState<string>('');
  const { userCredentials } = useCredentials();
  const [scheme, setScheme] = useState<Scheme>({});
  const [newScheme, setNewScheme] = useState<Scheme>({});

  // 3. 添加 viewType 状态
  const [viewType, setViewType] = useState<'graph' | 'chart'>('graph');

  // 4. 定义新的下拉选项
  const vizOptions: OptionType[] = [
    { label: '节点及其相关性分数图', value: 'graph' },
    { label: '剪枝前后节点和关系数对比图', value: 'chart' },
  ];

  // 3. 更新 dropdownVal 的初始值
  const [dropdownVal, setDropdownVal] = useState<OptionType>(vizOptions[0]);

  // 8. 移除 handleCheckboxChange

  const handleZoomToFit = () => {
    // 仅在图表视图时才缩放
    if (viewType === 'graph') {
      nvlRef.current?.fit(
        allNodes.map((node) => node.id),
        {}
      );
    }
  };

  // Destroy the component
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      handleZoomToFit();
    }, 10);
    return () => {
      nvlRef.current?.destroy();
      // setGraphType(intitalGraphType); // 8. 移除
      clearTimeout(timeoutId);
      setScheme({});
      setNodes([]);
      setRelationships([]);
      setAllNodes([]);
      setAllRelationships([]);
      setDropdownVal(vizOptions[0]); // 8. 更新
      setViewType('graph'); // 8. 添加
    };
  }, []);

  // To get nodes and relations on basis of view
  // 假设此 API 调用或传入的 props (nodeValues) 已经是 "剪枝后" 的数据
  const fetchData = useCallback(async () => {
    try {
      const nodeRelationshipData =
        viewPoint === 'showGraphView'
          ? await graphQueryAPI(
              userCredentials as UserCredentials,
              graphQuery,
              selectedRows?.map((f) => f.name)
            )
          : await graphQueryAPI(userCredentials as UserCredentials, graphQuery, [inspectedName ?? '']);
      return nodeRelationshipData;
    } catch (error: any) {
      console.log(error);
    }
  }, [viewPoint, selectedRows, graphQuery, inspectedName, userCredentials]);

  // Api call to get the nodes and relations
  const graphApi = async () => {
    try {
      const result = await fetchData();
      if (result && result.data.data.nodes.length > 0) {
        // 这些是 "剪枝后" 的数据
        const neoNodes = result.data.data.nodes.map((f: Node) => f);
        const neoRels = result.data.data.relationships.map((f: Relationship) => f);
        const { finalNodes, finalRels, schemeVal } = processGraphData(neoNodes, neoRels);
        setAllNodes(finalNodes); // 剪枝后
        setAllRelationships(finalRels); // 剪枝后
        setScheme(schemeVal);
        setNodes(finalNodes); // 剪枝后
        setRelationships(finalRels); // 剪枝后
        setNewScheme(schemeVal);
        setLoading(false);
      } else {
        setLoading(false);
        setStatus('danger');
        setStatusMessage(`No Nodes and Relations for the ${inspectedName} file`);
      }
    } catch (error: any) {
      setLoading(false);
      setStatus('danger');
      setStatusMessage(error.message);
    }
  };

  useEffect(() => {
    if (open) {
      setLoading(true);
      if (viewPoint !== 'chatInfoView') {
        graphApi();
      } else {
        // 假设 nodeValues 和 relationshipValues 也是 "剪枝后" 的数据
        const { finalNodes, finalRels, schemeVal } = processGraphData(nodeValues ?? [], relationshipValues ?? []);
        setAllNodes(finalNodes);
        setAllRelationships(finalRels);
        setScheme(schemeVal);
        setNodes(finalNodes);
        setRelationships(finalRels);
        setNewScheme(schemeVal);
        setLoading(false);
      }
    }
  }, [open]);

  if (!open) {
    return <></>;
  }

  const headerTitle = 'Alignment Relationship';
  const dropDownView = viewPoint !== 'chatInfoView';

  const nvlCallbacks = {
    onLayoutComputing(isComputing: boolean) {
      if (!isComputing) {
        handleZoomToFit();
      }
    },
  };

  const handleZoomIn = () => {
    nvlRef.current?.setZoom(nvlRef.current.getScale() * 1.3);
  };

  const handleZoomOut = () => {
    nvlRef.current?.setZoom(nvlRef.current.getScale() * 0.7);
  };

  const handleRefresh = () => {
    graphApi();
    // setGraphType(intitalGraphType); // 8. 移除
    setDropdownVal(vizOptions[0]); // 8. 更新
    setViewType('graph'); // 8. 添加
  };

  const onClose = () => {
    setStatus('unknown');
    setStatusMessage('');
    setGraphViewOpen(false);
    setScheme({});
    // setGraphType(intitalGraphType); // 8. 移除
    setNodes([]);
    setRelationships([]);
    setDropdownVal(vizOptions[0]); // 8. 更新
    setViewType('graph'); // 8. 添加
  };

  const legendCheck = Object.keys(newScheme).sort((a, b) => {
    if (a === 'Document' || a === 'Chunk') {
      return -1;
    } else if (b === 'Document' || b === 'Chunk') {
      return 1;
    }
    return a.localeCompare(b);
  });

  // 8. 移除 getDropdownDefaultValue
  // 8. 移除 initGraph

  // 5. 更新 handleDropdownChange
  const handleDropdownChange = (selectedOption: OptionType | null | void) => {
    if (selectedOption?.value) {
      setViewType(selectedOption.value as 'graph' | 'chart');
      setDropdownVal(selectedOption);
    }
  };

  // 7. 定义条形图数据
  const barChartData = [
    {
      name: 'Nodes (节点)',
      'Before Pruning (剪枝前)': nodes.length,
      'After Pruning (剪枝后)': 81, // 'nodes' state holds pruned nodes
    },
    {
      name: 'Relationships (关系)',
      'Before Pruning (剪枝前)': relationships.length,
      'After Pruning (剪枝后)': 133, // 'relationships' state holds pruned rels
    },
  ];

  // 6. 为图形定义自定义 nvlOptions 以显示分数
  const customNvlOptions = {
    ...nvlOptions, // 使用常量中的基础配置
    nodeTooltip: (node: Node) => {
      let tooltipContent = `<b>${node.labels.join(', ')}</b><br/>`;

      // 优先显示 name 或 id
      if (node.properties.name) {
        tooltipContent += `name: ${node.properties.name}<br/>`;
      } else if (node.properties.id) {
        tooltipContent += `id: ${node.properties.id}<br/>`;
      }

      // 显示相关性分数 (假设它在 properties.score 中)
      if (node.properties.score) {
        // 假设 score 是数字，格式化为4位小数
        const score = typeof node.properties.score === 'number'
          ? node.properties.score.toFixed(4)
          : node.properties.score;
        tooltipContent += `<b>score: ${score}</b><br/>`;
      }

      // 显示其他属性
      Object.entries(node.properties).forEach(([key, value]) => {
        if (key !== 'name' && key !== 'id' && key !== 'score') {
          tooltipContent += `${key}: ${value}<br/>`;
        }
      });
      return tooltipContent;
    },
  };

  return (
    <>
      <Dialog
        modalProps={{
          className: 'h-[90%]',
          id: 'default-menu',
        }}
        size='unset'
        open={open}
        aria-labelledby='form-dialog-title'
        disableCloseButton={false}
        onClose={onClose}
      >
        <Dialog.Header id='graph-title'>
          {headerTitle}
          <Flex className='w-full' alignItems='center' justifyContent='flex-end' flexDirection='row'>
            {/* 4. 更新 DropdownComponent */}
            {dropDownView && (
              <DropdownComponent
                onSelect={handleDropdownChange}
                options={vizOptions} // 使用新选项
                placeholder='Select View Type'
                // defaultValue={getDropdownDefaultValue()} // 8. 移除
                view='GraphViewModal.tsx'
                isDisabled={loading}
                value={dropdownVal}
              />
            )}
          </Flex>
        </Dialog.Header>
        <Dialog.Content className='flex flex-col n-gap-token-4 w-full grow overflow-auto border border-palette-neutral-border-weak'>
          <div className='bg-white relative w-full h-full max-h-full'>
            {loading ? (
              <div className='my-40 flex items-center justify-center'>
                <LoadingSpinner size='large' />
              </div>
            ) : status !== 'unknown' ? (
              <div className='my-40 flex items-center justify-center'>
                <Banner name='graph banner' description={statusMessage} type={status} />
              </div>
            ) : nodes.length === 0 && viewType === 'graph' ? ( // 5. 仅在图表视图且无数据时显示 "No Entities"
              <div className='my-40 flex items-center justify-center'>
                <Banner name='graph banner' description='No Entities Found' type='danger' />
              </div>
            ) : (
              <>
                {/* 5. 条件渲染 */}
                {viewType === 'graph' ? (
                  // 6. 渲染图形
                  <div className='flex' style={{ height: '100%' }}>
                    <div className='bg-palette-neutral-bg-default relative' style={{ width: '100%', flex: '1' }}>
                      <InteractiveNvlWrapper
                        nodes={nodes} // 剪枝后
                        rels={relationships} // 剪枝后
                        nvlOptions={customNvlOptions} // 6. 使用自定义选项
                        ref={nvlRef}
                        mouseEventCallbacks={{ ...mouseEventCallbacks }}
                        interactionOptions={{
                          selectOnClick: true,
                        }}
                        nvlCallbacks={nvlCallbacks}
                      />
                      <IconButtonArray orientation='vertical' floating className='absolute bottom-4 right-4'>
                        {viewPoint !== 'chatInfoView' && (
                          <IconButtonWithToolTip
                            label='Refresh'
                            text='Refresh graph'
                            onClick={handleRefresh}
                            placement='left'
                          >
                            <ArrowPathIconOutline />
                          </IconButtonWithToolTip>
                        )}
                        <IconButtonWithToolTip label='Zoomin' text='Zoom in' onClick={handleZoomIn} placement='left'>
                          <MagnifyingGlassPlusIconOutline />
                        </IconButtonWithToolTip>
                        <IconButtonWithToolTip
                          label='Zoom out'
                          text='Zoom out'
                          onClick={handleZoomOut}
                          placement='left'
                        >
                          <MagnifyingGlassMinusIconOutline />
                        </IconButtonWithToolTip>
                        <IconButtonWithToolTip
                          label='Zoom to fit'
                          text='Zoom to fit'
                          onClick={handleZoomToFit}
                          placement='left'
                        >
                          <FitToScreenIcon />
                        </IconButtonWithToolTip>
                      </IconButtonArray>
                    </div>
                    <Resizable
                      defaultSize={{
                        width: 400,
                        height: '100%',
                      }}
                      minWidth={230}
                      maxWidth='72%'
                      enable={{
                        top: false,
                        right: false,
                        bottom: false,
                        left: true,
                        topRight: false,
                        bottomRight: false,
                        bottomLeft: false,
                        topLeft: false,
                      }}
                      handleComponent={{ left: <DragIcon className='absolute top-1/2 h-6 w-6' /> }}
                      handleClasses={{ left: 'ml-1' }}
                    >
                      <div className='legend_div'>
                        <h4 className='py-4 pt-3 ml-2'>Result Overview (Pruned)</h4>
                        <div className='flex gap-2 flex-wrap ml-2'>
                          {legendCheck.map((key, index) => (
                            <LegendsChip key={index} title={key} scheme={newScheme} nodes={nodes} />
                          ))}
                        </div>
                      </div>
                    </Resizable>
                  </div>
                ) : (
                  // 7. 渲染条形图
                  <div style={{ width: '100%', height: '100%', padding: '2rem' }}>
                    <ResponsiveContainer width='100%' height='100%'>
                      <BarChart
                        data={barChartData}
                        margin={{
                          top: 20,
                          right: 30,
                          left: 20,
                          bottom: 5,
                        }}
                      >
                        <CartesianGrid strokeDasharray='3 3' />
                        <XAxis dataKey='name' />
                        <YAxis allowDecimals={false} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey='Before Pruning (剪枝前)' fill='#8884d8' />
                        <Bar dataKey='After Pruning (剪枝后)' fill='#82ca9d' />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            )}
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
};
export default JointlkViewModal;