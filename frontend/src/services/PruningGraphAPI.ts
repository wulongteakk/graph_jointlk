import axios from 'axios';
import { url } from '../utils/Utils';
import { UserCredentials } from '../types';

const pruningGraphQueryAPI = async (
  userCredentials: UserCredentials,
  query_type: string,
  question: string,
  use_jointlk?:boolean,
  document_names: (string | undefined)[] | undefined
) => {
  try {
    const formData = new FormData();
    formData.append('uri', userCredentials?.uri ?? '');
    formData.append('database', userCredentials?.database ?? '');
    formData.append('userName', userCredentials?.userName ?? '');
    formData.append('password', userCredentials?.password ?? '');
    formData.append('query_type', query_type ?? 'entities');
    formData.append('document_names', JSON.stringify(document_names));
    if (use_jointlk !== undefined) {
      formData.append('use_jointlk', String(use_jointlk));
    }

    const response = await axios.post(`${url()}/pruning_graph_query`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response;
  } catch (error) {
    console.log('Error Posting the Question:', error);
    throw error;
  }
};
export default pruningGraphQueryAPI;