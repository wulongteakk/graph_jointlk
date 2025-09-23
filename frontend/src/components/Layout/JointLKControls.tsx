
import React, { useState, useContext } from "react";
import { UserCredentialsContext } from "../../context/UserCredentials";
import { AlertContext } from "../../context/Alert";
import { pruneGraphAPI } from "../../services/GraphEnhanceAPI";
import { PruneResult } from "../../types";
import CustomButton from "../UI/CustomButton";

const JointLKControls: React.FC = () => {
  const { uri, userName, password, database } = useContext(
    UserCredentialsContext
  );
  const { setAlert } = useContext(AlertContext);
  const [threshold, setThreshold] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
  const [pruneResult, setPruneResult] = useState<PruneResult | null>(null);

  const handlePrune = async () => {
    if (!uri || !userName || !password) {
      setAlert({
        show: true,
        message: "Please connect to the database first.",
        type: "error",
      });
      return;
    }
    setIsLoading(true);
    setPruneResult(null);
    try {
      const result = await pruneGraphAPI(
        uri,
        userName,
        password,
        database,
        threshold
      );
      setPruneResult(result);
      setAlert({
        show: true,
        message: `Graph pruned successfully! ${result.pruned_nodes_count} nodes removed.`,
        type: "success",
      });
      // 触发一个自定义事件来刷新图谱
      window.dispatchEvent(new CustomEvent("refreshGraph"));
    } catch (error: any) {
      setAlert({
        show: true,
        message: `Pruning failed: ${error.message}`,
        type: "error",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4 border-b-2 border-gray-300 dark:border-gray-500">
      <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-white">
        JointLK Graph Optimization
      </h3>
      <div className="mb-4">
        <label
          htmlFor="threshold"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Pruning Threshold:{" "}
          <span className="font-bold text-blue-600 dark:text-blue-400">
            {threshold.toFixed(2)}
          </span>
        </label>
        <input
          type="range"
          id="threshold"
          min="0"
          max="1"
          step="0.05"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
        />
      </div>
      <CustomButton
        onClick={handlePrune}
        text={isLoading ? "Pruning..." : "Prune Graph"}
        isLoading={isLoading}
        className="w-full"
      />
      {pruneResult && (
        <div className="mt-4 p-3 text-xs border rounded-lg bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
          <h4 className="font-bold mb-2">Pruning Result:</h4>
          <p>Nodes: {pruneResult.nodes_before} → {pruneResult.nodes_after} ({pruneResult.pruned_nodes_count} removed)</p>
          <p>Relations: {pruneResult.relations_before} → {pruneResult.relations_after}</p>
        </div>
      )}
    </div>
  );
};

export default JointLKControls;