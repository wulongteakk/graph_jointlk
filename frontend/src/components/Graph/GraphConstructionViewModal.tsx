import { Banner, Button, Dialog, Flex, TextInput, Textarea } from '@neo4j-ndl/react';
import { useEffect, useMemo, useState } from 'react';
import { GraphViewModalProps } from '../../types';
import { getMainChainAnnotationAPI, submitMainChainAnnotationAPI } from '../../utils/FileAPI';
import { useCredentials } from '../../context/UserCredentials';

const GraphConstructionViewModal = ({ open, setGraphViewOpen, selectedRows }: GraphViewModalProps) => {
  const { userCredentials } = useCredentials();
  const selectedDoc = useMemo(() => (selectedRows && selectedRows.length ? selectedRows[0] : undefined), [selectedRows]);

  const [accidentType, setAccidentType] = useState('');
  const [finalCausalChain, setFinalCausalChain] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<'success' | 'danger' | 'info' | null>(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (!open || !selectedDoc || !userCredentials) return;
    setAccidentType(selectedDoc.reviewed_accident_type ?? '');
    setFinalCausalChain(selectedDoc.final_causal_chain ?? '');
    getMainChainAnnotationAPI(
      userCredentials,
      selectedDoc.doc_id,
      selectedDoc.name,
      selectedDoc.kg_scope,
      selectedDoc.kg_id
    )
      .then((res) => {
        if (res?.status === 'Success' && res?.data) {
          setAccidentType(res.data.reviewed_accident_type ?? selectedDoc.reviewed_accident_type ?? '');
          setFinalCausalChain(res.data.final_causal_chain ?? selectedDoc.final_causal_chain ?? '');
        }
      })
      .catch(() => undefined);
  }, [open, selectedDoc, userCredentials]);

  const onClose = () => {
    setGraphViewOpen(false);
    setStatus(null);
    setMessage('');
  };

  const onSubmit = async () => {
    if (!userCredentials || !selectedDoc) return;
    setLoading(true);
    setStatus(null);
    setMessage('');
    try {
      const res = await submitMainChainAnnotationAPI(userCredentials, {
        doc_id: selectedDoc.doc_id,
        fileName: selectedDoc.name,
        kg_scope: selectedDoc.kg_scope,
        kg_id: selectedDoc.kg_id,
        accident_type: accidentType,
        final_causal_chain: finalCausalChain,
      });
      if (res?.status === 'Success') {
        setStatus('success');
        setMessage('人工主链标注保存成功');
        setTimeout(() => onClose(), 500);
      } else {
        setStatus('danger');
        setMessage(res?.message ?? '保存失败');
      }
    } catch (e) {
      setStatus('danger');
      setMessage('保存失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} aria-labelledby='main-chain-annotation-title'>
      <Dialog.Header id='main-chain-annotation-title'>人工主链标注</Dialog.Header>
      <Dialog.Content>
        <Flex flexDirection='column' gap='4'>
          {status && message ? <Banner type={status}>{message}</Banner> : null}
          <TextInput
            label='事故类型（accident_type）'
            value={accidentType}
            onChange={(e) => setAccidentType(e.target.value)}
            placeholder='请输入事故类型'
          />
          <Textarea
            label='最终主因果链（final_causal_chain）'
            value={finalCausalChain}
            onChange={(e) => setFinalCausalChain(e.target.value)}
            placeholder='例如：设备失效 -> 泄漏 -> 点火 -> 爆炸'
            rows={5}
          />
        </Flex>
      </Dialog.Content>
      <Dialog.Footer>
        <Flex justifyContent='end' gap='2'>
          <Button onClick={onClose} disabled={loading}>
            取消
          </Button>
          <Button
            fill='filled'
            color='primary'
            onClick={onSubmit}
            loading={loading}
            disabled={!accidentType.trim() || !finalCausalChain.trim()}
          >
            提交标注
          </Button>
        </Flex>
      </Dialog.Footer>
    </Dialog>
  );
};

export default GraphConstructionViewModal;