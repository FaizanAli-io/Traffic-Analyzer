const Card = ({ children, className = "" }) => (
  <div
    className={`bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-6 hover:shadow-2xl transition-all duration-300 ${className}`}
  >
    {children}
  </div>
);

export default Card;
