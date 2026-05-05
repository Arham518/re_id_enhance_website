import { motion } from 'motion/react';
import { Target, Shield, Cpu, Users, Database, Zap, BookOpen, Fingerprint, Search } from 'lucide-react';
import { cn } from '../lib/utils';

export default function About() {
  const team = [
    { name: "Sharlet Khokhar", role: "Lead Researcher", initials: "SK" },
    { name: "Muhammad Arham", role: "System Architect", initials: "MA" },
    { name: "Muhammad Hashim Khan", role: "AI Engineer", initials: "HK" }
  ];

  const features = [
    {
      icon: <Search className="w-6 h-6" />,
      title: "Smart Classifier",
      desc: " An automated classifier categorizes incoming fingerprints as aged, labour-damaged, or normal, seamlessly routing them to the correct restoration pipeline."
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Targeted Restoration",
      desc: "Specialized models, including U-Net and Inpainting, intelligently restore faint ridges and reconstruct missing patches."
    },
    {
      icon: <Cpu className="w-6 h-6" />,
      title: "Minutiae Recovery",
      desc: "The system recovers critical minutiae points and ridge orientation, transforming unreadable degraded prints into clear, high-quality records."
    },
    {
      icon: <Database className="w-6 h-6" />,
      title: "Robust Datasets",
      desc: "Trained on massive biometric repositories (FVC, SOCOFing, NIST) for industry-leading generalization."
    }
  ];

  return (
    <div className="pt-40 pb-32 bg-background">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-3xl mb-32"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-secondary/20 text-secondary text-[10px] font-bold mb-6 tracking-widest uppercase">
            <BookOpen className="w-4 h-4" /> Our Mission
          </div>
          <h1 className="text-5xl md:text-7xl font-black mb-8 tracking-tighter leading-none">
            Ensuring no identity is rejected</h1>
          <p className="text-xl text-muted-foreground font-medium leading-relaxed italic border-l-4 border-secondary pl-8 ml-2">
             "Our goal is to overcome the limitations of current scanners by digitally restoring worn, cut, and aged fingerprints, granting seamless access to essential services"
          </p>
        </motion.div>

        {/* Problem & Solution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-24 items-start mb-40">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-16"
          >
            <div className="space-y-6">
              <span className="text-xs font-black tracking-[.4em] text-secondary uppercase">THE CHALLENGE</span>
              <h2 className="text-4xl font-extrabold tracking-tight">The Degradation Gap</h2>
              <p className="text-muted-foreground text-lg leading-relaxed font-medium">
                Standard Automated Fingerprint Identification Systems (AFIS) are significantly affected when the fingerprints are damaged, aged, or wounded. When faced with the faint ridges of the elderly or the cut and scarred fingerprints of manual labourers, traditional systems fail. This leads to frustrating false rejections in pension claims, airport security, and forensic analysis.</p>
            </div>
            <div className="space-y-6">
              <span className="text-xs font-black tracking-[.4em] text-secondary uppercase">the innovation</span>
              <h2 className="text-4xl font-extrabold tracking-tight">Degradation-Aware Enhancement</h2>
              <p className="text-muted-foreground text-lg leading-relaxed font-medium">
               Our innovation lies in a smart classification and enhancement pipeline. The system first automatically identifies the specific type of fingerprint damage. Then, instead of just applying a basic filter, specialized restoration algorithms reconstruct broken ridges and recover missing minutiae details, ensuring highly accurate biometric matching.</p>
            </div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="relative"
          >
            <div className="absolute -inset-4 bg-secondary/5 blur-3xl opacity-50" />
            <div className="glass p-4 rounded-[3rem] premium-shadow border rotate-1">
              <img
                src="https://cdn.prod.website-files.com/61845f7929f5aa517ebab941/64672278be2f9f9cdf75fc20_A%20fingerprint%20enrollment%20scene%20at%20the%20Aratek%20airport%20kiosk%20machine.jpg"
                alt="Technology"
                className="rounded-[2.5rem] shadow-inner grayscale"
                referrerPolicy="no-referrer"
              />
            </div>
            {/* Overlay card */}
            <div className="absolute -bottom-10 -right-6 glass p-6 rounded-3xl premium-shadow border max-w-[200px] hidden md:block animate-bounce-slow">
              <Fingerprint className="w-8 h-8 text-secondary mb-3" />
            </div>
          </motion.div>
        </div>

        {/* Key Features */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-40">
          {features.map((f, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="p-6 rounded-[1.5rem] bg-secondary/10 border hover:border-secondary/50 transition-all hover:premium-shadow group"
            >
              <div className="w-10 h-10 bg-background border rounded-xl flex items-center justify-center text-primary mb-4 group-hover:scale-110 transition-transform">
                {f.icon}
              </div>
              <h3 className="text-lg font-black mb-2 tracking-tighter">{f.title}</h3>
              <p className="text-muted-foreground text-xs leading-relaxed font-medium">{f.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Team Section */}
        <div className="text-center">
          <div className="mb-20">
            <h2 className="text-sm font-black tracking-[0.4em] text-secondary uppercase mb-4">the innovators</h2>
            <h3 className="text-3xl md:text-5xl font-black tracking-tight max-w-4xl mx-auto">Behind RE Touch ID – Aged & Labour Fingerprint Enhancement Software</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            {team.map((member, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="group"
              >
                <div className="relative mb-6">
                  <div className="w-32 h-32 bg-primary dark:bg-primary-foreground rounded-[2.5rem] flex items-center justify-center mx-auto transition-all group-hover:rotate-6 group-hover:scale-105 premium-shadow">
                    <span className="text-primary-foreground dark:text-primary text-3xl font-black">{member.initials}</span>
                  </div>
                  <div className="absolute -bottom-2 right-1/2 translate-x-1/2 bg-secondary text-white p-2 rounded-xl scale-0 group-hover:scale-100 transition-transform">
                    <Users className="w-4 h-4" />
                  </div>
                </div>
                <h4 className="text-2xl font-bold mb-1 tracking-tight">{member.name}</h4>
                <p className="text-secondary font-black text-[10px] uppercase tracking-[0.3em]">{member.role}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

