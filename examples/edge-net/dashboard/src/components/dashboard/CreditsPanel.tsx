import { useState } from 'react';
import { Card, CardBody, Button, Progress } from '@heroui/react';
import { motion } from 'framer-motion';
import {
  Coins,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  Wallet,
  TrendingUp,
  Play,
  Zap,
  Brain,
  Database,
  AlertCircle,
} from 'lucide-react';
import { useNetworkStore } from '../../stores/networkStore';
import { useCredits } from '../../hooks/useCredits';
import type { JobSubmission } from '../../services/creditsService';

export function CreditsPanel() {
  const { stats } = useNetworkStore();
  const {
    available,
    pending,
    earned,
    spent,
    transactions,
    jobs,
    earningRate,
    submitJob,
    canAfford,
    getJobCost,
    formatCredits,
    isEarning,
  } = useCredits();

  const [submittingJob, setSubmittingJob] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmitJob = async (type: JobSubmission['type']) => {
    setSubmittingJob(type);
    setError(null);
    try {
      await submitJob(type, { demo: true, timestamp: Date.now() });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Job submission failed');
    } finally {
      setSubmittingJob(null);
    }
  };

  const jobTypes: { type: JobSubmission['type']; label: string; icon: typeof Zap }[] = [
    { type: 'compute', label: 'Compute', icon: Zap },
    { type: 'inference', label: 'Inference', icon: Brain },
    { type: 'training', label: 'Training', icon: TrendingUp },
    { type: 'storage', label: 'Storage', icon: Database },
  ];

  const formatTime = (date: Date) => {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <div className="space-y-6">
      {/* Balance Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 border border-emerald-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <Wallet className="text-emerald-400" size={24} />
                <span className="text-xs text-emerald-400/70">Available</span>
              </div>
              <p className="text-3xl font-bold text-white">{available.toFixed(2)}</p>
              <p className="text-sm text-emerald-400 mt-1">Credits</p>
            </CardBody>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="bg-gradient-to-br from-amber-500/20 to-amber-600/10 border border-amber-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <Clock className="text-amber-400" size={24} />
                <span className="text-xs text-amber-400/70">Pending</span>
              </div>
              <p className="text-3xl font-bold text-white">{pending.toFixed(2)}</p>
              <p className="text-sm text-amber-400 mt-1">Credits</p>
            </CardBody>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="bg-gradient-to-br from-sky-500/20 to-sky-600/10 border border-sky-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <TrendingUp className="text-sky-400" size={24} />
                <span className="text-xs text-sky-400/70">Total Earned</span>
              </div>
              <p className="text-3xl font-bold text-white">{earned.toFixed(2)}</p>
              <p className="text-sm text-sky-400 mt-1">Credits</p>
            </CardBody>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="bg-gradient-to-br from-violet-500/20 to-violet-600/10 border border-violet-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <Coins className="text-violet-400" size={24} />
                <span className="text-xs text-violet-400/70">Net Balance</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {(earned - spent).toFixed(2)}
              </p>
              <p className="text-sm text-violet-400 mt-1">Credits</p>
            </CardBody>
          </Card>
        </motion.div>
      </div>

      {/* Earning Rate Indicator */}
      {isEarning && (
        <motion.div
          className="flex items-center gap-3 p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.35 }}
        >
          <div className="relative">
            <Zap className="text-emerald-400" size={24} />
            <span className="absolute -top-1 -right-1 w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-emerald-400">Actively Contributing</p>
            <p className="text-xs text-zinc-400">
              Earning {formatCredits(earningRate)} rUv/second
            </p>
          </div>
          <div className="text-right">
            <p className="text-lg font-bold text-emerald-400">{formatCredits(earningRate * 3600)}</p>
            <p className="text-xs text-zinc-500">rUv/hour</p>
          </div>
        </motion.div>
      )}

      {/* Job Submission */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold mb-4">Deploy Jobs</h3>
        <p className="text-sm text-zinc-400 mb-4">Use your credits to run compute tasks on the network</p>

        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 mb-4">
            <AlertCircle className="text-red-400" size={16} />
            <span className="text-sm text-red-400">{error}</span>
          </div>
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {jobTypes.map(({ type, label, icon: Icon }) => {
            const cost = getJobCost(type);
            const affordable = canAfford(type);
            return (
              <Button
                key={type}
                className={`flex flex-col items-center gap-2 p-4 h-auto ${
                  affordable
                    ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30 hover:bg-sky-500/30'
                    : 'bg-zinc-800/50 text-zinc-500 border border-zinc-700'
                }`}
                isDisabled={!affordable || submittingJob === type}
                onPress={() => handleSubmitJob(type)}
              >
                {submittingJob === type ? (
                  <Play className="animate-spin" size={24} />
                ) : (
                  <Icon size={24} />
                )}
                <span className="font-medium">{label}</span>
                <span className="text-xs opacity-70">{cost} rUv</span>
              </Button>
            );
          })}
        </div>

        {/* Active Jobs */}
        {jobs.filter(j => j.status === 'running' || j.status === 'pending').length > 0 && (
          <div className="mt-4 pt-4 border-t border-white/10">
            <h4 className="text-sm font-medium text-zinc-400 mb-2">Active Jobs</h4>
            <div className="space-y-2">
              {jobs.filter(j => j.status === 'running' || j.status === 'pending').map(job => (
                <div key={job.id} className="flex items-center gap-3 p-2 rounded bg-zinc-800/50">
                  <Play className="text-amber-400 animate-pulse" size={14} />
                  <span className="text-sm text-white">{job.type}</span>
                  <span className="text-xs text-zinc-500 ml-auto">{job.status}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>

      {/* Earning Progress */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.45 }}
      >
        <h3 className="text-lg font-semibold mb-4">Daily Earning Progress</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-zinc-400">Compute Contribution</span>
              <span className="text-emerald-400">45.8 / 100 TFLOPS</span>
            </div>
            <Progress
              value={45.8}
              maxValue={100}
              classNames={{
                indicator: 'bg-gradient-to-r from-emerald-500 to-cyan-500',
                track: 'bg-zinc-800',
              }}
            />
          </div>

          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-zinc-400">Tasks Completed</span>
              <span className="text-sky-400">89,432 / 100,000</span>
            </div>
            <Progress
              value={89.432}
              maxValue={100}
              classNames={{
                indicator: 'bg-gradient-to-r from-sky-500 to-violet-500',
                track: 'bg-zinc-800',
              }}
            />
          </div>

          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-zinc-400">Uptime Bonus</span>
              <span className="text-violet-400">{stats.uptime.toFixed(1)}%</span>
            </div>
            <Progress
              value={stats.uptime}
              maxValue={100}
              classNames={{
                indicator: 'bg-gradient-to-r from-violet-500 to-pink-500',
                track: 'bg-zinc-800',
              }}
            />
          </div>
        </div>
      </motion.div>

      {/* Recent Transactions */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Recent Transactions</h3>
          <Button size="sm" variant="flat" className="bg-white/5 text-zinc-400">
            View All
          </Button>
        </div>

        <div className="space-y-3">
          {transactions.map((tx) => (
            <div
              key={tx.id}
              className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`p-2 rounded-full ${
                    tx.type === 'earn' ? 'bg-emerald-500/20' : 'bg-red-500/20'
                  }`}
                >
                  {tx.type === 'earn' ? (
                    <ArrowUpRight className="text-emerald-400" size={16} />
                  ) : (
                    <ArrowDownRight className="text-red-400" size={16} />
                  )}
                </div>
                <div>
                  <p className="text-sm font-medium text-white">{tx.reason}</p>
                  <p className="text-xs text-zinc-500">{formatTime(tx.timestamp)}</p>
                </div>
              </div>
              <span
                className={`font-semibold ${
                  tx.type === 'earn' ? 'text-emerald-400' : 'text-red-400'
                }`}
              >
                {tx.type === 'earn' ? '+' : ''}{tx.amount.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
