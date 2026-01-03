/**
 * useCredits Hook - Real-time credits management
 */

import { useState, useEffect, useCallback } from 'react';
import {
  creditsService,
  type CreditsState,
  type CreditTransaction,
  type JobSubmission,
} from '../services/creditsService';
import { useNetworkStore } from '../stores/networkStore';

export function useCredits() {
  const [state, setState] = useState<CreditsState>(creditsService.getState());
  const [transactions, setTransactions] = useState<CreditTransaction[]>([]);
  const [jobs, setJobs] = useState<JobSubmission[]>([]);
  const [earningRate, setEarningRate] = useState(0);

  // Subscribe to network store for contribution settings
  const contributionEnabled = useNetworkStore(s => s.contributionSettings.enabled);
  const cpuLimit = useNetworkStore(s => s.contributionSettings.cpuLimit);
  const credits = useNetworkStore(s => s.credits);
  const firebasePeers = useNetworkStore(s => s.firebasePeers);

  // Initialize and sync credits
  useEffect(() => {
    creditsService.initialize({
      available: credits.available,
      earned: credits.earned,
      spent: credits.spent,
      pending: credits.pending,
    });
  }, []);

  // Sync from store
  useEffect(() => {
    creditsService.syncFromStore(credits);
  }, [credits]);

  // Update network peers
  useEffect(() => {
    creditsService.updateNetworkPeers(firebasePeers.length);
  }, [firebasePeers]);

  // Start/stop earning based on contribution
  useEffect(() => {
    if (contributionEnabled) {
      creditsService.startEarning(cpuLimit);
    } else {
      creditsService.stopEarning();
    }
  }, [contributionEnabled, cpuLimit]);

  // Subscribe to credits service
  useEffect(() => {
    const unsubscribe = creditsService.subscribe((newState) => {
      setState(newState);
      setTransactions(creditsService.getTransactions(20));
      setJobs(creditsService.getJobs(10));
      setEarningRate(creditsService.getEarningRate());
    });

    // Initial load
    setState(creditsService.getState());
    setTransactions(creditsService.getTransactions(20));
    setJobs(creditsService.getJobs(10));
    setEarningRate(creditsService.getEarningRate());

    return unsubscribe;
  }, []);

  // Submit a job
  const submitJob = useCallback(
    async (type: JobSubmission['type'], payload: unknown, customCredits?: number) => {
      return creditsService.submitJob(type, payload, customCredits);
    },
    []
  );

  // Check if we can afford a job
  const canAfford = useCallback(
    (type: JobSubmission['type'], customCredits?: number) => {
      return creditsService.canAfford(type, customCredits);
    },
    []
  );

  // Get job cost
  const getJobCost = useCallback((type: JobSubmission['type']) => {
    return creditsService.getJobCost(type);
  }, []);

  // Format credits for display
  const formatCredits = useCallback((amount: number) => {
    if (amount >= 1) {
      return amount.toFixed(2);
    } else if (amount >= 0.01) {
      return amount.toFixed(4);
    } else {
      return amount.toFixed(6);
    }
  }, []);

  return {
    // State
    available: state.available,
    pending: state.pending,
    earned: state.earned,
    spent: state.spent,
    transactions,
    jobs,
    earningRate,

    // Actions
    submitJob,
    canAfford,
    getJobCost,
    formatCredits,

    // Derived
    isEarning: contributionEnabled,
    balance: state.available,
  };
}
